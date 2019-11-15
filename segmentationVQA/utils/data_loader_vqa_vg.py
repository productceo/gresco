"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data


class VGQADataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, transform=None, has_responses=True,
                 max_examples=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            has_responses: Also contains the responses to the questions.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
        """
        self.dataset = dataset
        self.transform = transform
        self.has_responses = has_responses
        self.max_examples = max_examples

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.image_indices = annos['image_indices']
            self.images = annos['images']

        question = self.questions[index]
        answer = self.answers[index]
        image_index = self.image_indices[index]
        image = self.images[image_index]

        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        if self.transform is not None:
            image = self.transform(image)
        return image, question, answer, qlength.item(), alength.item()

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        annos = h5py.File(self.dataset, 'r')
        return annos['questions'].shape[0]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[3], reverse=True)
    images, questions, answers, _, alengths = zip(*data)
    images = torch.stack(images, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    aindices = np.flip(np.argsort(alengths), axis=0).copy()
    aindices = torch.Tensor(aindices).long()
    return images, questions, answers, aindices


def get_vqavg_loader(dataset, transform, batch_size,
                     shuffle=True, has_responses=True,
                     num_workers=1, max_examples=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset.

    Args:
        dataset: Location of annotations hdf5 file.
        transform: Transformations that should be applied to the images.
        batch_size: How many data points per batch.
        shuffle: Boolean that decides if the data should be returned in a
            random order.
        has_responses: Doesn't return answers if set to False.
        num_workers: Number of threads to use.
        max_examples: Used for debugging. Assumes that we have a
            maximum number of training examples.

    Returns:
        A torch.utils.data.DataLoader for custom engagement dataset.

    """
    vgqa = VGQADataset(dataset, has_responses=has_responses,
                       transform=transform, max_examples=max_examples)
    data_loader = torch.utils.data.DataLoader(dataset=vgqa,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
