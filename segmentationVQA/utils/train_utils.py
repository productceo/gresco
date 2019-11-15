"""Utility functions for training.
"""

import json
import torch
import torchtext
import os
import random
import sys


# ===========================================================
# Vocabulary.
# ===========================================================

class Vocabulary(object):
    """Keeps track of all the words in the vocabulary.
    """

    # Reserved symbols
    SYM_PAD = '<pad>'    # padding.
    SYM_SOQ = '<start>'  # Start of question.
    SYM_SOR = '<resp>'   # Start of response.
    SYM_EOS = '<end>'    # End of sentence.
    SYM_UNK = '<unk>'    # Unknown word.

    def __init__(self):
        """Constructor for Vocabulary.
        """
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.SYM_PAD)
        self.add_word(self.SYM_SOQ)
        self.add_word(self.SYM_SOR)
        self.add_word(self.SYM_EOS)
        self.add_word(self.SYM_UNK)

    def add_word(self, word):
        """Adds a new word and updates the total number of unique words.

        Args:
            word: String representation of the word.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def remove_word(self, word):
        """Removes a specified word & updates the total number of unique words.

        Args:
            word: String representation of the word.
        """
        if word in self.word2idx:
            self.word2idx.pop(word)
            self.idx2word.pop(self.idx)
            self.idx -= 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, location):
        with open(location, 'wb') as f:
            json.dump({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}, f)

    def load(self, location):
        with open(location, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.idx = data['idx']

    def tokens_to_words(self, tokens):
        """Converts tokens to vocab words.

        Args:
            tokens: 1D Tensor of Token outputs.

        Returns:
            A list of words.
        """
        words = []
        for token in tokens:
            word = self.idx2word[str(token.item())]
            if word == self.SYM_EOS:
                break
            if word not in [self.SYM_PAD, self.SYM_SOQ,
                            self.SYM_SOR, self.SYM_EOS]:
                words.append(word)
        sentence = str(' '.join(words).encode('utf-8'))
        return sentence


def get_glove_embedding(name, embed_size, vocab):
    """Construct embedding tensor.

    Args:
        name (str): Which GloVe embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.

    Returns:
        embedding (vocab_size, embed_size): Tensor of
            GloVe word embeddings.
    """
    f = open(os.devnull, 'w')
    temp = sys.stdout
    sys.stdout = f

    glove = torchtext.vocab.GloVe(name=name,
                                  dim=str(embed_size))

    sys.stdout = temp
    f.close()
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        embedding[i] = glove[vocab.idx2word[str(i)]]
    return embedding


# ===========================================================
# Helpers.
# ===========================================================

def pad_sequences(sequences, max_length=None):
    """Convert the list of sequences into a tensor.

    Args:
        words: A list of indices for each word in a Vocabulary.
        max_length: The max_length of the generated tensor.

    Returns:
        A tensor of num_sequences X max_length.
    """
    lengths = [len(seq) for seq in sequences]
    if max_length is None:
        max_length = max(lengths)
    tensor = torch.zeros(len(sequences), max_length).long()
    for i, seq in enumerate(sequences):
        end = min(lengths[i], max_length)
        tensor[i, :end] = torch.from_numpy(seq[:end])
    return tensor


def bow_batch(tensor, max_length=None):
    """Converts the indices tensor into a BOW tensor.

    Args:
        tensor: A tensor of batch X indices.
        options: Useful things that don't matter.
        max_length: The max_length of the generated tensor.

    Returns:
        A tensor of batch X max_length.
    """
    lengths = process_lengths(tensor)
    tensor = tensor.data.cpu()
    if max_length is None:
        max_length = lengths.max().data[0]
    bags = torch.zeros(tensor.size(0), max_length)
    bags.scatter_(1, tensor, 1.0)
    bags[:, 0] = 0
    return bags


def bow(words, length, vocab):
    """Converts the words in a tensor into BOW array.

    Args:
        words: A tensor of indices to the vocab.
        length: The length of the words.
        vocab: A Vocabulary object.
    """
    bag = torch.zeros(len(vocab))
    bag[torch.LongTensor(words[1:length-1])] = 1.0
    return bag


def process_lengths(inputs, pad=0):
    """Calculates the lenght of all the sequences in inputs.

    Args:
        inputs: A batch of tensors containing the question or response
            sequences.

    Returns: A list of their lengths.
    """
    max_length = inputs.size(1)
    if inputs.size(0) == 1:
        lengths = list(max_length - inputs.data.eq(pad).sum(1))
    else:
        lengths = list(max_length - inputs.data.eq(pad).sum(1).squeeze())
    return lengths


def compute_mask(inputs, pad=0):
    """Calculates masks for all inputs.

    Args:
        inputs: A batch of tensors containing the question or response
            sequences.

    Returns: A padding mask
    """
    mask = torch.ne(inputs, pad).float()
    return mask


# ===========================================================
# Evaluation metrics.
# ===========================================================

def accuracy_at(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k.

    Disclaimer: Only seems to work binary predictions.

    Args:
        output: The predictions from the model.
        target: The ground truth labels.
        topk: For what k values do you want accuracy results for.

    Returns:
        A list of accuracies for each k.
    """
    output = torch.round(output).long()
    target = target.long()
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy(output, target):
    """Calculates the accuracy of the predictions.

    Args:
        output: LongTensor of model predictions.
        target: LongTensor of ground truth labels.

    Returns:
        The accuracy.
    """
    output = output.squeeze()
    target = target.long()
    correct = output.eq(target).sum().float()
    return correct / target.numel()


def precision(output, target, eps=10e-8):
    """Calculates the precision of the predictions.

    Args:
        output: The predictions from the model.
        target: The ground truth labels.
    Returns:
        The precision.
    """
    output = torch.round(output).squeeze()
    tp = torch.matmul(output, target).float()
    return tp / (output.sum().float() + eps)


def recall(output, target, eps=10e-8):
    """Calculates the recall of the predictions.
    Args:
        output: The predictions from the model.
        target: The ground truth labels.
    Returns:
        The recall.
    """
    output = torch.round(output).squeeze()
    tp = torch.matmul(output, target).float()
    return tp / (target.sum().float() + eps)


def gaussian_KL_loss(mus, logvars, eps=1e-8):
    """Calculates KL distance of mus and logvars from unit normal.

    Args:
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.

    Returns:
        KL loss between mus and logvars and the normal unit gaussian.
    """
    KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
    return KLD/(mus.size(0) + eps)


def uniform_kl_loss(zs, latent_size, eps=1e-8):
    """Calculates KL distance of latent variables from uniform distribution.

    Args:
        zs: Latent variables of size (batch_size x num_variables x latent_dim).
        latent_size: Number of latent categories.

    Returns:
        KL loss from uniform distribution.,
    """
    batch_size, num_variables, _ = zs.size()
    log_zs = torch.log(zs * latent_size + eps)
    entropy = torch.sum(torch.mul(log_zs, zs))/batch_size/num_variables
    return entropy


def vae_loss(outputs, targets, mus, logvars, criterion):
    """VAE loss that combines cross entropy with KL divergence.

    Args:
        outputs: The predictions made by the model.
        targets: The ground truth indices in the vocabulary.
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.
        criterion: The cross entropy criterion.
    """
    CE = criterion(outputs, targets)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = gaussian_KL_loss(mus, logvars)
    return CE + KLD


def parse_outputs(outputs, questions, answers, vocab):
    """Converts the model's outputs to actual words.

    Args:
        outputs: The outputs from the model. Contains vocab indices.
        answers: Ground truth answers.
        vocab: The Vocabulary instance used by the model.

    Returns:
        generated: List of batch_size model outputs in words.
        ytrue: List of batch_size answers.
    """
    qs, generated, ytrue = [], [], []
    sequence = torch.stack(outputs['sequence']).squeeze(2).t()
    slength = outputs['length']
    for i in range(answers.size(0)):
        length = slength[i]
        out = sequence[i][:length]
        tgt_id_seq = [out[di] for di in range(length)]
        output = vocab.tokens_to_words(tgt_id_seq[:-1])
        question = vocab.tokens_to_words(questions[i])
        answer = vocab.tokens_to_words(answers[i])
        ytrue.append(answer)
        qs.append(question)
        generated.append(output)

    return qs, generated, ytrue


def compare_outputs(outputs, questions, answers, logging, num_show=10):
    """Sanity check generated output as we train.

    Args:
        outputs: String of parsed model output.
        questions: String containing questions.
        answers: String containing answers.
        logging: logging to use to report results.
        num_show: Number of samples to show.
    """
    for _ in range(num_show):
        i = random.randint(0, len(outputs)-1)
        logging.info('Question: %s \t Model output: %s \t Target: %s'
                     % (questions[i], outputs[i], answers[i]))
