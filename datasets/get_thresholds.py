import glob
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import softmax
import sys

# python3 get_thresholds.py 0.8 gta2city_soft_labels

p = float(sys.argv[1])
npy_dir = sys.argv[2]
save_path = "./{}_cls2prob.pickle".format(sys.argv[2])
output_path = "./{}_thresholds_p{}.npy".format(sys.argv[2], p)
ignore_label = 250

if not os.path.exists(save_path):
    cls2prob = {}
    files = glob.glob(os.path.join(npy_dir, "*.npy"))
    for i, npy_file in enumerate(files):
        if i % 100 == 0:
            print("i: {}/ {}".format(i, len(files)))
        f = np.load(npy_file) #[c, h, w]
        f = softmax(f, axis=0)
        classes = f.argmax(0) #[h, w]
        prob = f.max(0) #[h, w]
        for c in np.unique(classes):
            if c not in cls2prob:
                cls2prob[c] = []
            cls2prob[c].extend(prob[classes == c])
    for c in cls2prob:
        cls2prob[c].sort(reverse=True)
    # with open(save_path, "wb+") as f:
    #     pickle.dump(cls2prob, f)
else:
    with open(save_path, "rb") as f:
        cls2prob = pickle.load(f)

class_list = ["road","sidewalk","building","wall",
                "fence","pole","traffic_light","traffic_sign","vegetation",
                "terrain","sky","person","rider","car",
                "truck","bus","train","motorcycle","bicycle"]

# print("p: {}".format(p))

thresholds = []
for c in range(len(cls2prob.keys())):
    prob_c = cls2prob[c]
    rank = int(p * len(prob_c))
    thresh = prob_c[rank]
    thresholds.append(thresh)
thresholds = np.array(thresholds)

for i in range(len(thresholds)):
    print("i: {}, class i: {}, thresh_i: {}".format(i, class_list[i], thresholds[i]))

np.save(output_path, thresholds)
