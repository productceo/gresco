from torchvision import transforms
from PIL import Image

import numpy as np
import torch


def overlay_image(object_image, scene_image, bbox, 
				  transform_img, threshold=50):
	y, x, h, w = bbox
	if h*w < threshold:
		i_h, i_w, _ = scene_image.size()
		y, x, h, w = int(i_h * 0.25), int(i_w * 0.25), int(i_h * 0.5), int(i_w * 0.5)

	object_image = Image.fromarray(object_image)
	
	transform_bbox = transforms.Compose([
		transforms.Resize((h, w))])
	transform_img = transforms.Compose([
        transforms.Resize((224, 224))])

	object_image = transform_bbox(object_image)
	object_image = torch.Tensor(np.asarray(object_image, dtype='uint8'))
	object_mask = object_image > 0
	scene_image[y:y+h, x:x+w, :] = scence_image[y:y+h, x:x+w, :] - scene_image[y:y+h, x:x+w, :] * object_mask + object_image
	scene_image = Image.fromarray(scene_image.numpy().astype(np.uint8()))
	return transform_img(scene_image)
