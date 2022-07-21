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
NUM_STEPS = 93750
NUM_STEPS_STOP = 60000 # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESUME = './pretrained/sourceonly.pth' 
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'LS' #'Vanilla'

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
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("--gpus", type=str, default="0,1", help="selected gpus")
    parser.add_argument("--dist", action="store_true", help="DDP")
    parser.add_argument("--ngpus_per_node", type=int, default=1, help='number of gpus in each node')
    parser.add_argument("--print-every", type=int, default=20, help='output message every n iterations')

    parser.add_argument("--src_dataset", type=str, default="gta5", help='training source dataset')
    parser.add_argument("--tgt_dataset", type=str, default="cityscapes_train", help='training target dataset')
    parser.add_argument("--tgt_val_dataset", type=str, default="cityscapes_val", help='training target dataset')
    parser.add_argument("--noaug", action="store_true", help="augmentation")
    parser.add_argument('--resize', type=int, default=2200, help='resize long size')
    parser.add_argument("--clrjit_params", type=str, default="0.0,0.0,0.0,0.0", help='brightness,contrast,saturation,hue')
    parser.add_argument('--rcrop', type=str, default='896,512', help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')
    parser.add_argument('--src_rootpath', type=str, default='datasets/gta5')
    parser.add_argument('--tgt_rootpath', type=str, default='datasets/cityscapes')
    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--no_droplast', action='store_true')
    parser.add_argument('--used_save_pseudo', action='store_true')
    parser.add_argument('--pseudo_labels_folder', type=str, default='')
    parser.add_argument('--conf_bank_length', type=int, default=100000)
    parser.add_argument('--conf_p', type=float, default=0.8)
    
    parser.add_argument("--batch_size_val", type=int, default=4, help='batch_size for validation')
    parser.add_argument("--resume", type=str, default=RESUME, help='resume weight')
    parser.add_argument("--freeze_bn", action="store_true", help="augmentation")
    parser.add_argument("--lambda_adv_src", type=float, default=0.1, help='weight for loss_adv_src')
    parser.add_argument("--lambda_adv_tgt", type=float, default=0.01, help='weight for loss_adv_tgt')
    parser.add_argument("--hidden_dim", type=int, default=128, help='number of selected negative samples')
    parser.add_argument("--layer", type=int, default=1, help='separate from which layer')
    parser.add_argument("--lambda_st", type=float, default=0.1, help='weight for loss_st')
    return parser.parse_args()


args = get_arguments()


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


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
            feature_extractor_weights = resume_weight['model_state_dict']
            head_weights = resume_weight['head_state_dict']
            classifier_weights = resume_weight['classifier_state_dict']
            feature_extractor_weights = {k.replace("module.", ""):v for k,v in feature_extractor_weights.items()}
            head_weights = {k.replace("module.", ""):v for k,v in head_weights.items()}
            classifier_weights = {k.replace("module.", ""):v for k,v in classifier_weights.items()}

        if gpu == 0:
            logger.info("freeze_bn: {}".format(args.freeze_bn))
        model = resnet_feature_extractor('resnet101', 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', freeze_bn=args.freeze_bn)
        if args.resume:
            model.load_state_dict(feature_extractor_weights)
        
        if args.layer == 0:
            model_B1 = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool)
        elif args.layer == 1:
            model_B1 = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool, model.backbone.layer1)
        elif args.layer == 2:
            model_B1 = nn.Sequential(model.backbone.conv1, model.backbone.bn1, model.backbone.relu, model.backbone.maxpool, model.backbone.layer1, model.backbone.layer2)

        model = resnet_feature_extractor('resnet101', 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', freeze_bn=args.freeze_bn)
        if args.resume:
            model.load_state_dict(feature_extractor_weights)
        
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
        
        model_D1 = FCDiscriminator(ndf, ndf=32)
        model_D2 = FCDiscriminator(args.num_classes, ndf=64)

        classifier = ASPP_Classifier_Gen(2048, [6, 12, 18, 24], [6, 12, 18, 24], args.num_classes, hidden_dim=args.hidden_dim)
        head, classifier = classifier.head, classifier.classifier
        if args.resume:
            head.load_state_dict(head_weights)
            classifier.load_state_dict(classifier_weights)
        
        aux_classifier = ASPP_Classifier_Gen(2048, [6, 12, 18, 24], [6, 12, 18, 24], args.num_classes, hidden_dim=args.hidden_dim)
        _, aux_classifier = aux_classifier.head, aux_classifier.classifier
        if args.resume:
            aux_classifier.load_state_dict(classifier_weights)

    model_B1.train()
    model_B2.train()
    model_B.train()
    model_D1.train()
    model_D2.train()
    head.train()
    classifier.train()
    aux_classifier.train()

    # cudnn.benchmark = True
    if gpu == 0:
        logger.info(model_B1)
        logger.info(model_B2)
        logger.info(model_B)
        logger.info(model_D1)
        logger.info(model_D2)
        logger.info(head)
        logger.info(classifier)
        logger.info(aux_classifier)
    else:
        logger = None

    if gpu == 0:
        logger.info("args.noaug: {}, args.resize: {}, args.rcrop: {}, args.hflip: {}, args.noshuffle: {}, args.no_droplast: {}".format(args.noaug, args.resize, args.rcrop, args.hflip, args.noshuffle, args.no_droplast))
    args.rcrop = [int(x.strip()) for x in args.rcrop.split(",")]
    args.clrjit_params = [float(x) for x in args.clrjit_params.split(',')]
 
    datasets = create_dataset(args, logger)

    # define optimizer
    model_params = [{'params': list(model_B1.parameters()) + list(model_B2.parameters()) + list(model_B.parameters())},
                    {'params': list(head.parameters()) + list(classifier.parameters()) + \
                        list(aux_classifier.parameters()), 'lr': args.learning_rate * 10}]
    optimizer = optim.SGD(model_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    assert len(optimizer.param_groups) == 2
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    # define model
    model_B1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B1)
    model_B1 = torch.nn.parallel.DistributedDataParallel(model_B1.cuda(), device_ids=[gpu], find_unused_parameters=True)

    model_B2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B2)
    model_B2 = torch.nn.parallel.DistributedDataParallel(model_B2.cuda(), device_ids=[gpu], find_unused_parameters=True)

    model_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B)
    model_B = torch.nn.parallel.DistributedDataParallel(model_B.cuda(), device_ids=[gpu], find_unused_parameters=True)

    head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head)
    head = torch.nn.parallel.DistributedDataParallel(head.cuda(), device_ids=[gpu], find_unused_parameters=True)

    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.parallel.DistributedDataParallel(classifier.cuda(), device_ids=[gpu], find_unused_parameters=True)

    model_D1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_D1)
    model_D1 = torch.nn.parallel.DistributedDataParallel(model_D1.cuda(), device_ids=[gpu], find_unused_parameters=True)

    model_D2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_D2)
    model_D2 = torch.nn.parallel.DistributedDataParallel(model_D2.cuda(), device_ids=[gpu], find_unused_parameters=True)

    aux_classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aux_classifier)
    aux_classifier = torch.nn.parallel.DistributedDataParallel(aux_classifier.cuda(), device_ids=[gpu], find_unused_parameters=True)

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
        if gpu == 0:
            logger.info("use LS-GAN")
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    interp = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard and gpu == 0:
        writer = SummaryWriter(args.snapshot_dir)

    if gpu == 0:
        logger.info("args.lambda_adv_src: {}, args.lambda_adv_tgt: {}".format(args.lambda_adv_src, args.lambda_adv_tgt))

    # validate(model_B2, model_B, head, classifier, seg_loss, gpu, logger if gpu == 0 else None, datasets.target_valid_loader)
    # exit()

    trainloader_iter = enumerate(datasets.source_train_loader)
    targetloader_iter = enumerate(datasets.target_train_loader)

    conf_bank = {i: [] for i in range(args.num_classes)}
    thresholds = torch.zeros(args.num_classes).float().cuda()
    class_list = ["road","sidewalk","building","wall",
                    "fence","pole","traffic_light","traffic_sign","vegetation",
                    "terrain","sky","person","rider","car",
                    "truck","bus","train","motorcycle","bicycle"]

    scaler = torch.cuda.amp.GradScaler()
    best_miou = 0.0
    filename = None
    epoch_s, epoch_t = 0, 0
    for i_iter in range(args.num_steps):

        # model.train()
        model_B1.train()
        model_B2.train()
        model_B.train()
        model_D1.train()
        model_D2.train()
        head.train()
        classifier.train()
        aux_classifier.train()

        loss_seg_value = 0
        loss_adv_src_value = 0
        loss_adv_tgt_value = 0
        loss_D1_value = 0
        loss_D2_value = 0
        loss_st_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D1.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source
            try:
                _, batch = trainloader_iter.__next__()
            except StopIteration:
                epoch_s += 1
                datasets.source_train_sampler.set_epoch(epoch_s)
                trainloader_iter = enumerate(datasets.source_train_loader)
                _, batch = trainloader_iter.__next__()
                
            images = batch['img'].cuda()
            labels = batch['label'].cuda()

            src_size = images.shape[-2:]
            with torch.cuda.amp.autocast():
                feat_src = model_B1(images)

                feat_B_src = model_B(feat_src)
                pred = classifier(head(feat_B_src))
                pred = interp(pred) #[b, num_classes, h, w]

                temperature = 1.8
                pred = pred.div(temperature)
                loss_seg = seg_loss(pred, labels)
                
                D_out = model_D1(F.interpolate(feat_src, size=src_size, mode='bilinear', align_corners=True))

                loss_adv_src = args.lambda_adv_src * bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda())
                loss = loss_seg + loss_adv_src

                # proper normalization
                loss = loss / args.iter_size
                loss_seg_value += loss_seg / args.iter_size
                loss_adv_src_value += loss_adv_src / args.iter_size

            scaler.scale(loss).backward()

            # train with target
            try:
                _, batch = targetloader_iter.__next__()
            except StopIteration:
                epoch_t += 1
                datasets.target_train_sampler.set_epoch(epoch_t)
                targetloader_iter = enumerate(datasets.target_train_loader)
                _, batch = targetloader_iter.__next__()
                
            images = batch['img'].cuda()

            tgt_size = images.shape[-2:]
            with torch.cuda.amp.autocast():
                feat_tgt = model_B2(images)
                feat_B_tgt = model_B(feat_tgt)

                feat_B_tgt_head = head(feat_B_tgt)
                pred_tgt = classifier(feat_B_tgt_head)

                with torch.no_grad():
                    pred_logits, pred_idx = F.softmax(pred_tgt.detach(), 1).max(1) #[b, h, w]
                    assert pred_logits.shape[-2:] == pred_tgt.shape[-2:]

                    # update_thresholds
                    for c in range(args.num_classes):
                        prob_c = pred_logits[pred_idx == c].cpu().numpy().tolist()
                        if len(prob_c) == 0:
                            continue
                        conf_bank[c].extend(prob_c)
                        rank = int(len(conf_bank[c]) * args.conf_p)
                        thresholds[c] = sorted(conf_bank[c], reverse=True)[rank]
                        if len(conf_bank[c]) > args.conf_bank_length:
                            conf_bank[c] = conf_bank[c][-args.conf_bank_length:]
                    
                    n = torch.tensor(1.0).cuda()
                    dist.all_reduce(thresholds)
                    dist.all_reduce(n)
                    thresholds = thresholds / n
                    
                    if i_iter % 500 == 0 and gpu == 0:
                        for c in range(args.num_classes):
                            print("c: {}, class_i: {} threshold: {}, len(conf_bank[c]): {}".format(c, class_list[c], thresholds[c], len(conf_bank[c])))
                    
                    # if i_iter % 100 == 0 and gpu == 0:
                    #     num_pos = (pred_logits > thresholds[pred_idx]).float().sum()
                    #     num_all = np.prod(pred_logits.shape)
                    #     ratio = num_pos / (num_all+1e-8)
                    #     logger.info("num_pos: {}, num_all: {}, ratio: {}".format(num_pos, num_all, ratio))

                    pred_idx[pred_logits < thresholds[pred_idx]] = args.ignore_label

                pred_tgt = interp_target(pred_tgt)
                pred_tgt = pred_tgt.div(temperature)

                pred_tgt_aux = aux_classifier(feat_B_tgt_head)
                loss_st = args.lambda_st * seg_loss(pred_tgt_aux, pred_idx)

                D_out = model_D2(F.softmax(pred_tgt, 1))

                loss_adv_tgt = args.lambda_adv_tgt * bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
                loss = loss_adv_tgt + loss_st

                loss = loss / args.iter_size
                loss_adv_tgt_value += loss_adv_tgt / args.iter_size
                loss_st_value += loss_st / args.iter_size

            scaler.scale(loss).backward()

            # train D
            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            optimizer_D1.zero_grad()
            with torch.cuda.amp.autocast():
                src_D1_pred = model_D1(F.interpolate(feat_src.detach(), size=src_size, mode='bilinear', align_corners=True))
                loss_D1_src = 0.5 * bce_loss(src_D1_pred, torch.FloatTensor(src_D1_pred.data.size()).fill_(source_label).cuda()) / args.iter_size
            
            scaler.scale(loss_D1_src).backward()

            with torch.cuda.amp.autocast():
            
                tgt_D1_pred = model_D1(F.interpolate(feat_tgt.detach(), size=tgt_size, mode='bilinear', align_corners=True))
                loss_D1_tgt = 0.5 * bce_loss(tgt_D1_pred, torch.FloatTensor(tgt_D1_pred.data.size()).fill_(target_label).cuda()) / args.iter_size

                loss_D1_value += loss_D1_src + loss_D1_tgt

            scaler.scale(loss_D1_tgt).backward()

            for param in model_D2.parameters():
                param.requires_grad = True
            optimizer_D2.zero_grad()

            with torch.cuda.amp.autocast():
                src_D2_pred = model_D2(F.softmax(pred.detach(), 1))
                loss_D2_src = 0.5 * bce_loss(src_D2_pred, torch.FloatTensor(src_D2_pred.data.size()).fill_(source_label).cuda()) / args.iter_size

            scaler.scale(loss_D2_src).backward()

            with torch.cuda.amp.autocast():
                    
                tgt_D2_pred = model_D2(F.softmax(pred_tgt.detach(), 1))
                loss_D2_tgt = 0.5 * bce_loss(tgt_D2_pred, torch.FloatTensor(tgt_D2_pred.data.size()).fill_(target_label).cuda()) / args.iter_size
                
                loss_D2_value += loss_D2_src + loss_D2_tgt

            scaler.scale(loss_D2_tgt).backward()

        n = torch.tensor(1.0).cuda()

        dist.all_reduce(n), dist.all_reduce(loss_seg_value), dist.all_reduce(loss_adv_src_value), dist.all_reduce(loss_adv_tgt_value)
        dist.all_reduce(loss_D1_value), dist.all_reduce(loss_D2_value), dist.all_reduce(loss_st_value)
        
        loss_seg_value = loss_seg_value.item() / n.item()
        loss_adv_src_value = loss_adv_src_value.item() / n.item()
        loss_adv_tgt_value = loss_adv_tgt_value.item() / n.item()
        loss_D1_value = loss_D1_value.item() / n.item()
        loss_D2_value = loss_D2_value.item() / n.item()
        loss_st_value = loss_st_value.item() / n.item()

        scaler.step(optimizer)
        scaler.step(optimizer_D1)
        scaler.step(optimizer_D2)
        scaler.update()

        if args.tensorboard and gpu == 0:
            scalar_info = {
                'loss_seg': loss_seg_value,
                'loss_adv_src': loss_adv_src_value,
                'loss_adv_tgt': loss_adv_tgt_value,
                'loss_D1': loss_D1_value,
                'loss_D2': loss_D2_value,
                "loss_st": loss_st_value,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        if gpu == 0 and i_iter % args.print_every == 0:
            logger.info('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_src = {3:.5f}, loss_adv_tgt = {4:.5f}, loss_D1 = {5:.3f}, '
                'loss_D2 = {6:.3f}, loss_st = {7:.5f}, epoch_s = {8:3d}, epoch_t = {9:3d}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_src_value, \
                loss_adv_tgt_value, loss_D1_value, loss_D2_value, loss_st_value, epoch_s, epoch_t))
        
        if gpu == 0 and i_iter >= args.num_steps_stop - 1:
            logger.info('save model ...')
            filename = osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth')
            save_file = {'model_B1_state_dict': model_B1.state_dict(), 'model_B2_state_dict': model_B2.state_dict(), \
                    'model_B_state_dict': model_B.state_dict(), 'head_state_dict': head.state_dict(), 'classifier_state_dict': classifier.state_dict()}
            torch.save(save_file, filename)
            logger.info("saving checkpoint model to {}".format(filename))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            miou, loss_val = validate(model_B2, model_B, head, classifier, seg_loss, gpu, logger if gpu == 0 else None, datasets.target_valid_loader)
            if args.tensorboard and gpu == 0:
                scalar_info = {
                    'miou_val': miou,
                    'loss_val': loss_val
                }
                for k, v in scalar_info.items():
                    writer.add_scalar(k, v, i_iter)

            if gpu == 0 and miou > best_miou:
                best_miou = miou
                logger.info('taking snapshot ...')
                if filename is not None and os.path.exists(filename):
                    os.remove(filename)
                filename = osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + "_{}".format(miou) + '.pth')
                save_file = {'model_B1_state_dict': model_B1.state_dict(), 'model_B2_state_dict': model_B2.state_dict(), \
                        'model_B_state_dict': model_B.state_dict(), 'head_state_dict': head.state_dict(), 'classifier_state_dict': classifier.state_dict()}
                torch.save(save_file, filename)
                logger.info("saving checkpoint model to {}".format(filename))
                
    if args.tensorboard and gpu == 0:
        writer.close()

def validate(model_B2, model_B, head, classifier, seg_loss, gpu, logger, testloader):
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
            images = batch['img'].cuda()
            labels = batch['label'].cuda()
            
            pred = model_B(model_B2(images))
            pred = classifier(head(pred))
            output = F.interpolate(pred, size=labels.size()[-2:], mode='bilinear', align_corners=True)
            loss = seg_loss(output, labels)
            
            output = output.max(1)[1]
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

    torch.cuda.empty_cache()

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
    if args.dist:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}" 
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args.dist_url))
    else:
        main_worker(args.train_gpu, args.world_size, args)

