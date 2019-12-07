"""Sampling script to sample a Segmentation VQA model.
"""
import os
import json
import nltk
import logging
import torch

from PIL import Image
from torchvision import transforms

from segmentationVQA.utils import Dict2Obj
from segmentationVQA.utils import load_vocab
from segmentationVQA.utils import process_lengths
from segmentationVQA.utils import get_glove_embedding
from segmentationVQA.models import SEGModel


def load_seg_model(model_path, vocab_path=''):
    """Loads the appropriate model.

    Args:
        model_path: model path.
        vocab_path: vocab file path.

    Returns:
        Returns the model and the vocab.
    """
    # Grab all the training parameters.
    model_dir = os.path.dirname(model_path)
    args = Dict2Obj(json.load(
            open(os.path.join(model_dir, "args.json"), "r")))
    # Load vocabulary wrapper.
    vocab = load_vocab(vocab_path)

    # Load GloVe embedding.
    embedding = None
    if args.use_glove:
        embedding = get_glove_embedding(
                args.embedding_name, args.vocab_embed_size, vocab)

    # Build the models
    logging.info("Building SEGVQA models...")
    segvqa = SEGModel(len(vocab), args.max_length, args.hidden_size,
                      args.vocab_embed_size,
                      vocab(vocab.SYM_SOQ),
                      vocab(vocab.SYM_EOS),
                      rnn_cell=args.rnn_cell, num_layers=args.num_layers,
                      bidirectional=args.bidirectional,
                      dropout_p=args.dropout,
                      input_dropout_p=args.dropout,
                      embedding=embedding)

    # Load the trained model parameters
    segvqa.load_state_dict(torch.load(model_path))

    # Wrap the decoder in TopKDecoder.
    segvqa.eval()

    if torch.cuda.is_available():
        segvqa = segvqa.cuda()

    return segvqa, vocab


def sample_segmentation_VQA(image, question, answer):
    """Print out predicted answer for single IQ pair.

    Args:
        image (str): Path to image.
        question (str): Natural language question to be answered.
        beam_size (int): Beam size to use.
    Returns:
        predicted answer.
    """
    segvqa, vocab = load_seg_model(
        "segmentationVQA/weights/segvqa1/segvqa-52.pkl",
        vocab_path='segmentationVQA/data/vocab_vqa_multi.json'
    )

    # Image preprocessing
    transform_im = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Prepare Image
    image = image.convert("RGB")
    image_test = transform_im(image).unsqueeze(0)
    image_tensor = transform(image).unsqueeze(0)

    # Prepare question
    tokens = nltk.tokenize.word_tokenize(str(question).lower())
    question = [vocab('<start>')]
    question.extend([vocab(token) for token in tokens])
    question.append(vocab('<end>'))
    while len(question) != 20:
        question.append(vocab('<pad>'))
    question_tensor = torch.Tensor(question).long().unsqueeze(0)
    #
    tokens = nltk.tokenize.word_tokenize(str(answer).lower())
    answer = [vocab('<start>')]
    answer.extend([vocab(token) for token in tokens])
    answer.append(vocab('<end>'))
    while len(question) != 20:
        answer.append(vocab('<pad>'))
    answer_tensor = torch.Tensor(answer).long().unsqueeze(0)

    # If use gpu
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        answer_tensor = answer_tensor.cuda()
        question_tensor = question_tensor.cuda()

    alengths = process_lengths(answer_tensor)

    # Forward.
    logits = segvqa(image_tensor, answer_tensor, alengths=alengths, questions=question_tensor)

    _, prediction = logits[0].squeeze(0).max(0)
    segmentation = prediction.data.cpu().numpy().squeeze()
    segmentation[segmentation > 0] = 1
    # object_without_scene = (image_test.data.cpu().numpy().squeeze() * segmentation)
    original_image = image_test.data.cpu().numpy().squeeze()
    object_without_scene = original_image * segmentation * 255
    image_with_mask = original_image + (segmentation*255 - object_without_scene)
    return segmentation, object_without_scene, image_with_mask, original_image
