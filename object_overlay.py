from torchvision import transforms
from PIL import Image

import numpy as np
import torch


def sample_object_overlay(object_image, scene_image, bounding_box, threshold=10):
    y, x, h, w = bounding_box
    scene_image = torch.Tensor(np.asarray(scene_image, dtype='uint8'))
    # print("h1, w1: {}".format((h, w)))

    if h < threshold or w < threshold:
        i_h, i_w, _ = scene_image.size()
        y, x, h, w = int(i_h * 0.25), int(i_w * 0.25), int(i_h * 0.5), int(i_w * 0.5)

    obj_width, obj_height = object_image.size
    scale = min(w/obj_width, h/obj_height)
    h = int(scale * obj_height)
    w = int(scale * obj_width)

    transform_img = transforms.Compose([transforms.Resize((224, 224))])
    transform_bbox = transforms.Compose([transforms.Resize((h, w))])
    # print("scene_image: {}".format(scene_image.size()))
    # print("h2, w2: {}".format((h, w)))
    # print("object_image: {}".format(object_image.size))
    object_image = transform_bbox(object_image)
    object_image = torch.Tensor(np.asarray(object_image, dtype='uint8'))
    if len(object_image.shape) == 2:
        object_image = object_image.unsqueeze(2)
    elif object_image.shape[2] > 3:
        object_image = object_image[:,:,:3]
    object_mask = object_image > 0

    # print(bounding_box)
    # print("y, x, h, w: {}".format((y, x, h, w)))
    # print("scene_image[y:y+h, x:x+w, :]: {}".format(scene_image[y:y+h, x:x+w, :].size()))
    # print("object_image: {}".format(object_image.size()))
    # print("NEWobject_mask: {}".format(object_mask.size()))
    scene_image[y:y+h, x:x+w, :] = scene_image[y:y+h, x:x+w, :] - scene_image[y:y+h, x:x+w, :] * object_mask + object_image
    scene_image = Image.fromarray(scene_image.numpy().astype(np.uint8()))
    return transform_img(scene_image)