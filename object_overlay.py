from torchvision import transforms
from PIL import Image

import numpy as np
import torch


def sample_object_overlay(object_image, scene_image, bounding_box, threshold=50):
    y, x, h, w = bounding_box
    scene_image = torch.Tensor(np.asarray(scene_image, dtype='uint8'))
    if h * w < threshold:
        i_h, i_w, _ = scene_image.size()
        y, x, h, w = int(i_h * 0.25), int(i_w * 0.25), int(i_h * 0.5), int(i_w * 0.5)

    obj_width, obj_height = object_image.size
    if obj_width > obj_height:
        h = int(w * (obj_height / obj_width))
    else:
        w = int(h * (obj_width / obj_height))
    transform_bbox = transforms.Compose([transforms.Resize((h, w))])
    transform_img = transforms.Compose([transforms.Resize((224, 224))])

    object_image = transform_bbox(object_image)
    object_image = torch.Tensor(np.asarray(object_image, dtype='uint8'))
    object_mask = object_image > 0
    scene_image[y:y+h, x:x+w, :] = scene_image[y:y+h, x:x+w, :] - scene_image[y:y+h, x:x+w, :] * object_mask + object_image
    scene_image = Image.fromarray(scene_image.numpy().astype(np.uint8()))
    return transform_img(scene_image)