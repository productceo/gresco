"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data


class VGQADataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, transform=None,
                 max_examples=None, indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: The indices that should be used from the dataset.
        """
        self.dataset = dataset
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
            self.image_masks = annos['image_mask']

        # Grab the real index from the indices list.
        if self.indices is not None:
            index = self.indices[index]

        question = self.questions[index]
        answer = self.answers[index]
        image_mask = self.image_masks[index]
        image_index = self.image_indices[index]
        image = self.images[image_index]

        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        image_mask = torch.from_numpy(image_mask)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        if self.transform is not None:
            image = self.transform(np.uint8(image))
        return image, question, answer, image_mask, alength.item()

    def __len__(self):
        if self.max_examples is not None and self.indices is not None:
            return min(len(self.indices), self.max_examples)
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        return annos['questions'].shape[0]


def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[4], reverse=True)
    images, questions, answers, image_masks, _ = zip(*data)
    images = torch.stack(images, 0)
    image_masks = torch.stack(image_masks, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    return images, questions, answers, image_masks


def get_vg_loader(dataset, transform, batch_size, num_workers=6,
                  shuffle=True, max_examples=None, indices=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset.
    """
    vgqa = VGQADataset(
            dataset,
            transform=transform,
            max_examples=max_examples,
            indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=vgqa,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
