from h5py import File
from PIL import Image

from os import listdir
from os.path import join

from object_masking import sample_object_masking
from object_removal import sample_object_removal
from object_overlay import sample_object_overlay

from object_masking import train_object_masking
from object_removal import train_object_removal


class Gresco():
    def __init__(self, scene_images_dir, object_images_dir, dataset):
        self.dataset = File(dataset, "r")
        self.object_images = dict()
        self.scene_images = [join(scene_images_dir, f) for f in listdir(scene_images_dir)]
        for object_class in listdir(object_images_dir):
            object_class_dir = join(object_images_dir, object_class)
            self.object_images[object_class] = [
                join(object_class_dir, f) for f in listdir(object_class_dir)
            ]

    def generate_images_r(self, image, question, answer):
        (object_mask, object_without_scene, image_with_mask, bounding_box) = sample_object_masking(image, question, answer)
        for scene_image in self.scene_images:
            image_r = sample_object_overlay(object_without_scene, scene_image, object_mask)

    def generate_dataset_r(self):
        for index in range(len(self.dataset['answers'])):
            image_index = self.dataset['image_indices'][index]
            image = self.dataset['images'][image_index]
            question = self.dataset['questions'][index]
            answer = self.dataset['answers'][index]
            self.generate_images_r(image, question, answer)

    def test_sample_object_overlay(self, image, question, answer, object_image):
        object_image = Image.open(object_image)
        object_mask, object_without_scene, image_with_mask, bounding_box, original_image = sample_object_masking(image, question, answer)
        scene_without_object = sample_object_removal(original_image, object_mask)
        scene_with_object = sample_object_overlay(object_image, scene_without_object, bounding_box)
        return scene_without_object, image_with_mask, scene_with_object

    def train(self):
        train_object_masking()
