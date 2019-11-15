import os
import numpy as np
from segmentationVQA import sample_segmentation_VQA


def get_bounding_box(object_mask):
    i, j = np.where(object_mask == 1)
    if len(i) == 0:
        return (0, 0, 0, 0)
    else:
        y = i[0]
        x = j[0]
        height = i[-1] - y
        width = j[-1] - y
        return (y, x, height, width)


def sample_object_masking(image, question, answer):
    object_mask, object_without_scene, image_with_mask = sample_segmentation_VQA(image, question, answer)
    bounding_box = get_bounding_box(object_mask)
    return (object_mask, object_without_scene, image_with_mask, bounding_box)


def train_object_masking():
    os.system("cd segmentation-VQA")
    os.system("python train_segmentation_VQA.py")
    os.system("cd ..")
    return
