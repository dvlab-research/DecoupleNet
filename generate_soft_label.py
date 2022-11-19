import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import logging
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from model.feature_extractor import resnet_feature_extractor
from model.classifier import ASPP_Classifier_Gen
from model.discriminator import FCDiscriminator

from utils.util import *
from data import create_dataset
import cv2

IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMG_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 16
IGNORE_LABEL = 250
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 62500
NUM_STEPS_STOP = 40000 # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESUME = './pretrained/model_phase1.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

SET = 'train'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gpus", type=str, default="0,1", help="selected gpus")
    parser.add_argument("--dist", action="store_true", help="DDP")
    parser.add_argument("--ngpus_per_node", type=int, default=1, help='number of gpus in each node')
    parser.add_argument("--print-every", type=int, default=20, help='output message every n iterations')

    parser.add_argument("--src_dataset", type=str, default="gta5", help='training source dataset')
    parser.add_argument("--tgt_dataset", type=str, default="cityscapes_train", help='training target dataset')
    parser.add_argument("--tgt_val_dataset", type=str, default="cityscapes_val", help='training target dataset')
    parser.add_argument("--noaug", action="store_true", help="augmentation")
    parser.add_argument('--resize', type=int, default=2200, help='resize long size')
    parser.add_argument("--clrjit_params", type=str, default="0.5,0.5,0.5,0.2", help='brightness,contrast,saturation,hue')
    parser.add_argument('--rcrop', type=str, default='896,512', help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')
    parser.add_argument('--src_rootpath', type=str, default='datasets/gta5')
    parser.add_argument('--tgt_rootpath', type=str, default='datasets/cityscapes')
    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--no_droplast', action='store_true')
    parser.add_argument('--pseudo_labels_folder', type=str, default='')
    parser.add_argument("--batch_size_val", type=int, default=4, help='batch_size for validation')
    parser.add_argument("--resume", type=str, default=RESUME, help='resume weight')
    parser.add_argument("--freeze_bn", action="store_true", help="augmentation")
    parser.add_argument("--hidden_dim", type=int, default=128, help='number of selected negative samples')
    parser.add_argument("--layer", type=int, default=1, help='separate from which layer')
    parser.add_argument("--output_folder", type=str, default="", help='output folder')
    return parser.parse_args()


args = get_arguments()

def main_worker(gpu, world_size, dist_url):
    """Create the model and start the training."""
    if gpu == 0:
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
        logFilename = os.path.join(args.snapshot_dir, str(time.time()))
        logging.basicConfig(
                        level = logging.INFO,
                        format ='%(asctime)s-%(levelname)s-%(message)s',
                        datefmt = '%y-%m-%d %H:%M',
                        filename = logFilename,
                        filemode = 'w+')
        filehandler = logging.FileHandler(logFilename, encoding='utf-8')
        logger = logging.getLogger()
        logger.addHandler(filehandler)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.info(args)

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.random_seed) # if you are using multi-GPU.
    # torch.backends.cudnn.enabled = False

    print("gpu: {}, world_size: {}".format(gpu, world_size))
    print("dist_url: ", dist_url)

    torch.cuda.set_device(gpu)
    args.batch_size = args.batch_size // world_size
    args.batch_size_val = args.batch_size_val // world_size
    args.num_workers = args.num_workers // world_size
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=gpu)

    if gpu == 0:
        logger.info("args.batch_size: {}, args.batch_size_val: {}".format(args.batch_size, args.batch_size_val))

    device = torch.device("cuda" if not args.cpu else "cpu")
    args.world_size = world_size

    if gpu == 0:
        logger.info("args: {}".format(args))

    # cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        if args.resume:
            resume_weight = torch.load(args.resume, map_location='cpu')
            print("args.resume: ", args.resume)
            # feature_extractor_weights = resume_weight['model_state_dict']
            model_B2_weights = resume_weight['model_B2_state_dict']
            model_B_weights = resume_weight['model_B_state_dict']
            head_weights = resume_weight['head_state_dict']
            classifier_weights = resume_weight['classifier_state_dict']
            # feature_extractor_weights = {k.replace("module.", ""):v for k,v in feature_extractor_weights.items()}
            model_B2_weights = {k.replace("module.", ""):v for k,v in model_B2_weights.items()}
            model_B_weights = {k.replace("module.", ""):v for k,v in model_B_weights.items()}
            head_weights = {k.replace("module.", ""):v for k,v in head_weights.items()}
            classifier_weights = {k.replace("module.", ""):v for k,v in classifier_weights.items()}

        if gpu == 0:
            logger.info("freeze_bn: {}".format(args.freeze_bn))
        model = resnet_feature_extractor('resnet101', 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', freeze_bn=args.freeze_bn)

        if args.layer == 0:
            ndf = 64
            model_B2 = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool)
            model_B = nn.Sequential(model.backbone.layer1, model.backbone.layer2, model.backbone.layer3, model.backbone.layer4)
        elif args.layer == 1:
            ndf = 256
            model_B2 = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool, model.backbone.layer1)
            model_B = nn.Sequential(model.backbone.layer2, model.backbone.layer3, model.backbone.layer4)
        elif args.layer == 2:
            ndf = 512
            model_B2 = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool, model.backbone.layer1, model.backbone.layer2)
            model_B = nn.Sequential(model.backbone.layer3, model.backbone.layer4)
        
        if args.resume:
            model_B2.load_state_dict(model_B2_weights)
            model_B.load_state_dict(model_B_weights)

        classifier = ASPP_Classifier_Gen(2048, [6, 12, 18, 24], [6, 12, 18, 24], args.num_classes, hidden_dim=args.hidden_dim)
        head, classifier = classifier.head, classifier.classifier
        if args.resume:
            head.load_state_dict(head_weights)
            classifier.load_state_dict(classifier_weights)

    model_B2.train()
    model_B.train()
    head.train()
    classifier.train()

    if gpu == 0:
        logger.info(model_B2)
        logger.info(model_B)
        logger.info(head)
        logger.info(classifier)
    else:
        logger = None

    if gpu == 0:
        logger.info("args.noaug: {}, args.resize: {}, args.rcrop: {}, args.hflip: {}, args.noshuffle: {}, args.no_droplast: {}".format(args.noaug, args.resize, args.rcrop, args.hflip, args.noshuffle, args.no_droplast))
    args.rcrop = [int(x.strip()) for x in args.rcrop.split(",")]
    args.clrjit_params = [float(x) for x in args.clrjit_params.split(',')]

    datasets = create_dataset(args, logger)
    sourceloader_iter = enumerate(datasets.source_train_loader)
    targetloader_iter = enumerate(datasets.target_train_loader)

    # define model
    model_B2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B2)
    model_B2 = torch.nn.parallel.DistributedDataParallel(model_B2.cuda(), device_ids=[gpu], find_unused_parameters=True)

    model_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B)
    model_B = torch.nn.parallel.DistributedDataParallel(model_B.cuda(), device_ids=[gpu], find_unused_parameters=True)

    head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head)
    head = torch.nn.parallel.DistributedDataParallel(head.cuda(), device_ids=[gpu], find_unused_parameters=True)

    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.parallel.DistributedDataParallel(classifier.cuda(), device_ids=[gpu], find_unused_parameters=True)
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    interp = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard and gpu == 0:
        writer = SummaryWriter(args.snapshot_dir)
        
    validate(model_B2, model_B, head, classifier, seg_loss, gpu, logger if gpu == 0 else None, datasets.target_train_loader, args.output_folder)
    # exit()

def validate(model_B2, model_B, head, classifier, seg_loss, gpu, logger, testloader, output_folder):
    if gpu == 0:
        logger.info("Start Evaluation")
    # evaluate
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model_B2.eval()
    model_B.eval()
    head.eval()
    classifier.eval()

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            images = batch["img_full"].cuda()
            labels = batch["lbl_full"].cuda()
            img_paths = batch['img_path']

            pred = model_B(model_B2(images))
            pred = classifier(head(pred))
            output = F.interpolate(pred, size=labels.size()[-2:], mode='bilinear', align_corners=True)
            loss = seg_loss(output, labels)
            
            output = F.softmax(output, 1)

            output_np = pred.detach().cpu().numpy().squeeze()

            logits, output = output.max(1)

            for b in range(output_np.shape[0]):
                mask_filename = img_paths[b].split("/")[-1].split(".")[0]
                np.save(os.path.join(output_folder, mask_filename+".npy"), output_np[b])

            intersection, union, _ = intersectionAndUnionGPU(output, labels, args.num_classes, args.ignore_label)
            dist.all_reduce(intersection), dist.all_reduce(union)
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union)
            loss_meter.update(loss.item(), images.size(0))
            if gpu == 0 and i % 50 == 0 and i != 0:
                logger.info("Evaluation iter = {0:5d}/{1:5d}, loss_eval = {2:.3f}".format(
                    i, len(testloader), loss_meter.val
                ))
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    miou = np.mean(iou_class)
    if gpu == 0:
        logger.info("Val result: mIoU = {:.3f}".format(miou))
        for i in range(args.num_classes):
            logger.info("Class_{} Result: iou = {:.3f}".format(i, iou_class[i]))
        logger.info("End Evaluation")

    return miou, loss_meter.avg

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

if __name__ == '__main__':
    args.gpus = [int(x) for x in args.gpus.split(",")]
    args.world_size = len(args.gpus)

    os.makedirs(args.output_folder, exist_ok=True)

    if args.dist:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}" 
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args.dist_url))
    else:
        main_worker(args.train_gpu, args.world_size, args)

