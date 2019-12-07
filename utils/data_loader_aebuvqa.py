"""Loads data and feeds it to the models.
"""

import h5py
import torch
import torch.utils.data as data
import numpy as np


class VGQAstart(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, answer_feats_dataset,
                 index_remap=None, max_examples=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions, answers and image features.
            answer_feats_dataset: hdf5 file with answer features.
            index_remap: Containing a mapping from the index queries to the
                actual index in the hdf5 file. Useful when we only want to
                iterate over a portion of the data.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
        """
        self.dataset = dataset
        self.answer_feats_dataset = answer_feats_dataset
        self.index_remap = index_remap
        self.max_examples = max_examples

    def __getitem__(self, index):
        """Returns a triple (image, questions and answers).
        """
        if not hasattr(self, 'boxes'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.image_indices = annos['image_indices']
            self.image_feats = annos['image_feats']
            answer_features = h5py.File(self.answer_feats_dataset, 'r')
            self.answer_feats = answer_features['answer_feats']

        # Remap the index.
        if self.index_remap is not None:
            index = self.index_remap[index]

        ans_feats = self.answer_feats[index]
        ans_feats = torch.from_numpy(ans_feats)
        question = self.questions[index]
        question = torch.from_numpy(question)
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        answer = self.answers[index]
        answer = torch.from_numpy(answer)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        image_index = self.image_indices[index]
        img_feats = self.image_feats[image_index]
        img_feats = torch.from_numpy(img_feats)
        return img_feats, ans_feats, answer, question, qlength.item(), alength.item(), index

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
    """Creates mini-batch tensors from the list of septuples
    (imgage feats, answer feats, answer, question, qlength, 
        alength and index).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of septuple:
            - image features.
            - answer features.
            - answers.
            - questions.
            - qlengths: questions lengths.
            - alengths: answer lengths.
            - index: indices list for sampling.

    Returns:
        images: torch tensor of shape (batch_size, 100, 2048).
        questions: torch tensor of shape (batch_size, padded_lengths).
        ans_feats: torch tensor of shape (batch_size , ?).
        aindices: torch tensor of shape (batch_size, ).
        answers: torch tensor of shape (batch_size, padded_length).
        index: list of selected indices.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[4], reverse=True)
    images, ans_feats, answers, questions, qlengths, alengths, index = zip(*data)
    images = torch.stack(images)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    ans_feats = torch.stack(ans_feats, 0)
    aindices = np.flip(np.argsort(alengths), axis=0).copy()
    aindices = torch.Tensor(aindices).long()
    return images, questions, ans_feats, answers, aindices, index


def get_aebuvqa_loader(dataset, answer_feats_dataset, batch_size,
                       index_remap=None, shuffle=False, num_workers=6,
                       max_examples=None):
    """Returns torch.utils.data.DataLoader for a custom dataset.
    """
    vgqa = VGQAstart(dataset, answer_feats_dataset,
                     index_remap=index_remap,
                     max_examples=max_examples)
    data_loader = torch.utils.data.DataLoader(dataset=vgqa,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_train)
    return data_loader
