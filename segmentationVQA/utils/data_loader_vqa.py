"""Loads VQA classification dataset.
"""

import h5py
import torch


class VQADataset(torch.utils.data.Dataset):
    """VQA Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, max_examples=None, indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: The indices that should be used from the dataset.
        """
        self.dataset = dataset
        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if not hasattr(self, 'questions'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.categories = annos['categories']
            self.image_indices = annos['image_indices']
            self.feats = annos['feats']

        # Grab the real index from the indices list.
        if self.indices is not None:
            index = self.indices[index]

        question = self.questions[index]
        category = self.categories[index]
        feat_index = self.image_indices[index]
        feat = self.feats[feat_index]

        question = torch.from_numpy(question)
        feat = torch.from_numpy(feat)
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        return feat, question, category, qlength.item()

    def __len__(self):
        if self.max_examples is not None and self.indices is not None:
            return min(len(self.indices), self.max_examples)
        elif self.max_examples is not None:
            return self.max_examples
        elif self.indices is not None:
            return len(self.indices)
        else:
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
    feats, questions, categories, _ = zip(*data)
    feats = torch.stack(feats, 0)
    questions = torch.stack(questions, 0).long()
    categories = torch.LongTensor(categories)
    return feats, questions, categories


def get_vqa_loader(dataset, batch_size, shuffle, num_workers,
                  max_examples=None, indices=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset.
    """
    vqa = VQADataset(dataset, max_examples=max_examples, indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=vqa,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
