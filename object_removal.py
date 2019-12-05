import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image
from shutil import copyfile
from edge_connect.src.config import Config
from edge_connect.src.edge_connect import EdgeConnect


INPUT_FOLDER = "edge_connect/datasets/inputs"
MASK_FOLDER = "edge_connect/datasets/masks"
OUTPUT_FOLDER = "edge_connect/datasets/outputs"


def get_config():
    config_path = os.path.join("edge_connect/checkpoints/places2", 'config.yml')
    if not os.path.exists(config_path):
        copyfile('edge_connect/config.yml.example', config_path)
    config = Config(config_path)
    config.MODE = 2
    config.MODEL = 3
    config.INPUT_SIZE = 0
    config.TEST_FLIST = INPUT_FOLDER
    config.TEST_MASK_FLIST = MASK_FOLDER
    config.RESULTS = OUTPUT_FOLDER
    return config


def get_model(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")
    cv2.setNumThreads(0)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    model = EdgeConnect(config)
    return model


def prepare_directories(image, object_mask):
    for folder in [INPUT_FOLDER, MASK_FOLDER, OUTPUT_FOLDER]:
        os.system("rm -rf {}".format(folder))
        os.system("mkdir {}".format(folder))
    image = image.transpose((1, 2, 0))
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    object_mask = np.expand_dims(object_mask, axis=2)
    object_mask = object_mask.reshape((224, 224)).astype('float32')
    object_mask = cv2.resize(object_mask, (256, 256), interpolation=cv2.INTER_AREA)
    image *= 255
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    object_mask *= 255
    object_mask = object_mask.astype(np.float32)
    cv2.imwrite('{}/temp.png'.format(INPUT_FOLDER), image)
    cv2.imwrite('{}/temp.png'.format(MASK_FOLDER), object_mask)


def sample_object_removal(image, object_mask):
    prepare_directories(image, object_mask)
    # config = get_config()
    # model = get_model(config)
    # model.load()
    # model.test()
    os.system("python edge_connect/test.py \
        --checkpoints edge_connect/checkpoints/places2 \
        --input {} \
        --mask {} \
        --output {}".format(INPUT_FOLDER, MASK_FOLDER, OUTPUT_FOLDER))
    output = os.listdir(OUTPUT_FOLDER)[0]
    return Image.open("{}/{}".format(OUTPUT_FOLDER, output))


def train_object_removal():
    os.system("sh prepare_scene_blending.sh")
