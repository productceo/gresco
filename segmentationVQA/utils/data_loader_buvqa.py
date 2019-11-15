"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data


class VGQAstart(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, features, index_remap=None, max_examples=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            features: the location of the image features.
            index_remap: Containing a mapping from the index queries to the
                actual index in the hdf5 file. Useful when we only want to
                iterate over a portion of the data.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
        """
        self.dataset = dataset
        self.features = features
        self.index_remap = index_remap
        self.max_examples = max_examples

    def __getitem__(self, index):
        """Returns a triple (image, questions and answers).
        """
        if not hasattr(self, 'questions'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.image_indices = annos['image_indices']
            feat_annos = h5py.File(self.features, 'r')
            self.feats = feat_annos['image_feats']

        # Remap the index.
        if self.index_remap is not None:
            index = self.index_remap[index]

        question = self.questions[index]
        answer = self.answers[index]
        feat_index = self.image_indices[index]
        feat = self.feats[feat_index]

        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        feat = torch.from_numpy(feat)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        return feat, question, answer, index, qlength.item(), alength.item()

    def __len__(self):
        if self.max_examples is not None and self.index_remap is not None:
            return min(len(self.index_remap), self.max_examples)
        elif self.max_examples is not None:
            return self.max_examples
        elif self.index_remap is not None:
            return len(self.index_remap)
        else:
            annos = h5py.File(self.dataset, 'r')
            return annos['questions'].shape[0]


def collate_train(data):
    """Creates mini-batch tensors from the list of quadruples
    (image, questions, answers and qlengths).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[4], reverse=True)
    feats, questions, answers, index, _, alengths = zip(*data)
    feats = torch.stack(feats, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    aindices = np.flip(np.argsort(alengths), axis=0).copy()
    aindices = torch.Tensor(aindices).long()
    return feats, questions, answers, aindices, index


def get_bu_loader(dataset, features, batch_size, max_examples=None,
                 index_remap=None, shuffle=False, num_workers=6):
    """Returns torch.utils.data.DataLoader for a custom dataset.
    """
    vgqa = VGQAstart(dataset, features, index_remap=index_remap,
                     max_examples=max_examples)
    data_loader = torch.utils.data.DataLoader(dataset=vgqa,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_train)
    return data_loader
