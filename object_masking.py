import os
import numpy as np
from segmentationVQA import sample_segmentation_VQA
from PIL import Image


def get_bounding_box(object_mask):
    i, j = np.where(object_mask == 1)
    i = np.sort(i)
    j = np.sort(j)
    # print(i[:5])
    # print(i[-5:])
    # print(j[:5])
    # print(j[-5:])
    if len(i) == 0 or len(j) == 0:
        return (0, 0, 0, 0)
    else:
        y = i[0]
        x = j[0]
        height = i[-1] - y
        width = j[-1] - x
        return (y, x, height, width)


def sample_object_masking(image, question, answer):
    object_mask, object_without_scene, image_with_mask, original_image = sample_segmentation_VQA(image, question, answer)
    object_without_scene = Image.fromarray(np.uint8(object_without_scene.transpose((1, 2, 0))))
    bounding_box = get_bounding_box(object_mask)

    (y, x, height, width) = bounding_box
    # if y+height > object_without_scene.size[0] or x+width > object_without_scene.size[1]:
    #     print("ERROR")
    #     print(bounding_box)
    #     print(object_mask.shape)
    #     print(object_without_scene.size)
    #     print(original_image.shape)

    return (object_mask, object_without_scene, image_with_mask, bounding_box, original_image)


def train_object_masking():
    os.system("cd segmentation-VQA")
    os.system("python train_segmentation_VQA.py")
    os.system("cd ..")
    return
