import torch
import torch.utils.data as data

import json
import random
import numpy as np

from h5py import File
from PIL import Image

import argparse
import progressbar

from os import listdir
from os.path import join

from object_masking import sample_object_masking
from object_removal import sample_object_removal
from object_overlay import sample_object_overlay

from object_masking import train_object_masking
from object_removal import train_object_removal

from utils import Vocabulary
from nltk.stem import WordNetLemmatizer 

class Gresco():
    def __init__(self, scene_images_dir, object_images_dir, dataset, vocab):
        self.last_index = 0
        self.vocab = Vocabulary()
        self.vocab.load(vocab)
        self.lemmatizer = WordNetLemmatizer() 
        self.dataset = File(dataset, 'r')
        self.questions = self.dataset['questions']
        self.answers = self.dataset['answers']
        self.image_indices = self.dataset['image_indices']
        self.images = self.dataset['images']
        # self.dataset_size = 100000
        self.dataset_size = self.questions.shape[0]
        self.object_images = dict()
        self.object_classes = []
        self.scene_images = dict()
        self.scene_classes = []
        for object_class in listdir(object_images_dir):
            self.object_classes.append(self.lemmatizer.lemmatize(object_class))
        for scene_class in listdir(scene_images_dir):
            self.scene_classes.append(self.lemmatizer.lemmatize(scene_class))
        for object_class in listdir(object_images_dir):
            object_class_dir = join(object_images_dir, object_class)
            self.object_images[object_class] = [
                join(object_class_dir, f) for f in listdir(object_class_dir)
            ]
        for scene_class in listdir(scene_images_dir):
            scene_class_dir = join(scene_images_dir, scene_class)
            self.scene_images[scene_class] = [
                join(scene_class_dir, f) for f in listdir(scene_class_dir)
            ]


    def generate_images_g(self, image, question, answer, object_class):
        samples_per_class = 3
        images_g = []
        object_masking_result = sample_object_masking(
                                                      image, 
                                                      question, 
                                                      answer
                                                      )
        object_mask = object_masking_result[0]
        object_without_scene = object_masking_result[1]
        image_with_mask = object_masking_result[2]
        bounding_box = object_masking_result[3]
        original_image = object_masking_result[4]

        if bounding_box[2] == 0 or bounding_box[3] == 0:
            return images_g

        scene_without_object = sample_object_removal(
                                                     original_image, 
                                                     object_mask
                                                     )
        object_images = self.object_images[object_class]
        random.shuffle(object_images)
        sample_count = 0
        for object_image in object_images:
            if sample_count == samples_per_class: break
            sample_count += 1
            object_image = Image.open(object_image)
            scene_with_object = sample_object_overlay(
                                            object_image, 
                                            scene_without_object, 
                                            bounding_box
                                            )
            filepath = "datasets/output/generalizability"
            self.last_index += 1
            image_index = self.last_index
            filename = "{}/gre_{}.jpg".format(filepath, str(image_index).zfill(8))
            scene_with_object.save(filename)
            images_g.append((image_index, question, answer))

        return images_g


    def generate_images_r(self, image, question, answer, object_class):
        samples_per_class = 3
        sample_classes = 3
        images_r = []
        object_masking_result = sample_object_masking(
                                                      image, 
                                                      question, 
                                                      answer
                                                      )
        object_mask = object_masking_result[0]
        object_without_scene = object_masking_result[1]
        image_with_mask = object_masking_result[2]
        bounding_box = object_masking_result[3]
        original_image = object_masking_result[4]

        if bounding_box[2] == 0 or bounding_box[3] == 0:
            return images_r
        
        scene_classes = self.scene_classes
        random.shuffle(scene_classes)
        sample_class_count = 0
        for scene_class in scene_classes:
            if sample_class_count == sample_classes: break
            sample_class_count += 1
            scene_images = self.scene_images[scene_class]
            random.shuffle(scene_images)
            sample_count = 0
            for scene_image in scene_images:
                if sample_count == samples_per_class: break
                sample_count += 1
                scene_image = Image.open(scene_image)
                scene_with_object = sample_object_overlay(
                                                object_without_scene, 
                                                scene_image, 
                                                bounding_box
                                                )
                filepath = "datasets/output/robustness"
                self.last_index += 1
                image_index = self.last_index
                filename = "{}/gre_{}.jpg".format(filepath, str(image_index).zfill(8))
                scene_with_object.save(filename)
                images_r.append((image_index, question, answer))

        return images_r


    def generate_images_e(self, image, question, answer, object_class):
        samples_per_class = 3
        sample_classes = 3
        images_e = []
        object_masking_result = sample_object_masking(
                                                      image, 
                                                      question, 
                                                      answer
                                                      )
        object_mask = object_masking_result[0]
        object_without_scene = object_masking_result[1]
        image_with_mask = object_masking_result[2]
        bounding_box = object_masking_result[3]
        original_image = object_masking_result[4]

        if bounding_box[2] == 0 or bounding_box[3] == 0:
            return images_e

        scene_without_object = sample_object_removal(
                                                     original_image, 
                                                     object_mask
                                                     )

        object_classes = self.object_classes
        random.shuffle(object_classes)
        sample_class_count = 0
        for new_object_class in object_classes:
            if sample_class_count == sample_classes: break
            sample_class_count += 1
            if new_object_class == object_class: continue
            object_images = self.object_images[new_object_class]
            random.shuffle(object_images)
            sample_count = 0
            for object_image in object_images:
                if sample_count == samples_per_class: break
                sample_count += 1
                object_image = Image.open(object_image)
                scene_with_object = sample_object_overlay(
                                                object_image, 
                                                scene_without_object, 
                                                bounding_box
                                                )
                filepath = "datasets/output/extensibility"
                self.last_index += 1
                image_index = self.last_index
                filename = "{}/gre_{}.jpg".format(filepath, str(image_index).zfill(8))
                scene_with_object.save(filename)
                images_e.append((image_index, question, new_object_class))

        return images_e


    def generate_dataset(self, start_index, end_index, gre_func, 
                         q_filename, a_filename, message=None):
        if message is not None: print(message)
        questions = dict()
        questions["questions"] = []
        annotations = dict()
        annotations["annotations"] = []
        bar = progressbar.ProgressBar(maxval=end_index - start_index)
        bar.start()
        count = 0
        for index in range(start_index, end_index):
            count += 1
            bar.update(count)
            image, question, answer = self.get_dataset_item(index)

            self.last_index += 1
            image_index = self.last_index
            image.save("datasets/output/generalizability/gre_{}.jpg".format(str(image_index).zfill(8)))
            image.save("datasets/output/robustness/gre_{}.jpg".format(str(image_index).zfill(8)))
            image.save("datasets/output/extensibility/gre_{}.jpg".format(str(image_index).zfill(8)))
            questions["questions"].append({
                'image_id': image_index,
                'question': question,
                'question_id': image_index
            })
            annotations["annotations"].append({
                'image_id': image_index,
                'question_type': 'object',
                'multiple_choice_answer': answer,
                'question_id': image_index,
                'answer_type': 'object',
                'answers': [{'answer': answer, 'answer_id': 1, 'answer_confidence': 'yes'}]
            })

            object_class = self.lemmatizer.lemmatize(answer)
            if object_class in self.object_classes:
                images_gre = gre_func(image, question, answer, object_class)
                for image_gre in images_gre:
                    image_index, question, answer = image_gre
                    questions["questions"].append({
                        'image_id': image_index,
                        'question': question,
                        'question_id': image_index
                    })
                    annotations["annotations"].append({
                        'image_id': image_index,
                        'question_type': 'object',
                        'multiple_choice_answer': answer,
                        'question_id': image_index,
                        'answer_type': 'object',
                        'answers': [{'answer': answer, 'answer_id': 1, 'answer_confidence': 'yes'}]
                    })
        json.dump(questions, open(q_filename, 'w'))
        json.dump(annotations, open(a_filename, 'w'))


    def generate_dataset_g(self):
        train_val_cutoff = int(self.dataset_size * 0.8)
        self.generate_dataset(
                             0, 
                             train_val_cutoff, 
                             self.generate_images_g,
                             "datasets/output/generalizability/train_questions.json",
                             "datasets/output/generalizability/train_annotations.json",
                             "Preparing Dataset G Train\n"
                             )
        self.generate_dataset(
                             train_val_cutoff, 
                             self.dataset_size, 
                             self.generate_images_g,
                             "datasets/output/generalizability/val_questions.json",
                             "datasets/output/generalizability/val_annotations.json",
                             "Preparing Dataset G Val\n"
                             )


    def generate_dataset_r(self):
        train_val_cutoff = int(self.dataset_size * 0.8)
        self.generate_dataset(
                             0, 
                             train_val_cutoff, 
                             self.generate_images_r,
                             "datasets/output/robustness/train_questions.json",
                             "datasets/output/robustness/train_annotations.json",
                             "Preparing Dataset R Train\n"
                             )
        self.generate_dataset(
                             train_val_cutoff, 
                             self.dataset_size, 
                             self.generate_images_r,
                             "datasets/output/robustness/val_questions.json",
                             "datasets/output/robustness/val_annotations.json",
                             "Preparing Dataset R Val\n"
                             )



    def generate_dataset_e(self):
        train_val_cutoff = int(self.dataset_size * 0.8)
        self.generate_dataset(
                             0, 
                             train_val_cutoff, 
                             self.generate_images_e,
                             "datasets/output/extensibility/train_questions.json",
                             "datasets/output/extensibility/train_annotations.json",
                             "Preparing Dataset E Train\n"
                             )
        self.generate_dataset(
                             train_val_cutoff, 
                             self.dataset_size, 
                             self.generate_images_e,
                             "datasets/output/extensibility/val_questions.json",
                             "datasets/output/extensibility/val_annotations.json",
                             "Preparing Dataset E Val\n"
                             )


    def get_dataset_item(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        image_index = self.image_indices[index]
        image = Image.fromarray(np.uint8(self.images[image_index]))
        question = self.vocab.tokens_to_words(question)
        answer = self.vocab.tokens_to_words(answer)
        return image, question, answer

    
    def generate_dataset_gre(self):
        self.generate_dataset_g()
        self.generate_dataset_r()
        self.generate_dataset_e()
        print("READY for build_gre_dataset.sh")


    def sample_gre(self, image, question, answer, object_class):
        images_g = self.generate_images_g(image, question, answer, object_class)
        images_r = self.generate_images_r(image, question, answer, object_class)
        images_e = self.generate_images_e(image, question, answer, object_class)
        return images_g, images_r, images_e


    def train(self):
        train_object_masking()

def main(args):
    gresco = Gresco(
        scene_images_dir = args.scene_images_dir,
        object_images_dir = args.object_images_dir,
        dataset = args.dataset,
        vocab = args.vocab
    )
    gresco.generate_dataset_gre()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-images-dir', type=str,
                        default='datasets/scenes')
    parser.add_argument('--object-images-dir', type=str,
                        default='datasets/objects')
    parser.add_argument('--dataset', type=str,
                        default='datasets/input/vqa_multi_dataset.hdf5')
    parser.add_argument('--vocab', type=str,
                        default='datasets/input/vocab_vqa_multi.json')
    args = parser.parse_args()
    main(args)
    
