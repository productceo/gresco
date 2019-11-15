"""Training script to train a Segmentation VQA model.
"""
import os
import time
import json
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import confusion_matrix

from segmentationVQA.models import SEGModel
from segmentationVQA.utils import get_vg_loader
from segmentationVQA.utils import get_glove_embedding
from segmentationVQA.utils import Vocabulary
from segmentationVQA.utils import load_vocab
from segmentationVQA.utils import process_lengths


def flatten_logits(logits, num_classes=2):
    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, num_classes)
    return logits_flatten


def get_miou(logits, target, cm=None):
    if cm is None:
        logits = logits.data
        _, prediction = logits.max(1)
        prediction = prediction.squeeze(1)
        prediction_np = prediction.cpu().numpy().flatten()
        annotation_np = target.cpu().numpy().flatten()
        cm = confusion_matrix(y_true=annotation_np,
                              y_pred=prediction_np,
                              labels=[0, 1])
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.astype(np.float32)
    miou = np.mean(iou)
    return miou, cm


def sample_image(logits, targets):
    i = np.random.choice(targets.size(0))
    _, prediction = logits[i].squeeze(0).max(0)
    segmentation = prediction.data.cpu().numpy().squeeze()
    segmentation[segmentation > 0] = 255
    segmentation = segmentation[:, :, None].repeat(3, axis=2)
    segmentation = segmentation.astype(np.uint8)
    target = targets[i, :, :, 0].data.cpu().numpy()
    target[target > 0] = 255
    target = target[:, :, None].repeat(3, axis=2)
    target = target.astype(np.uint8)
    image = np.concatenate((segmentation, target), axis=1)
    plt.imsave('data/pred.png', image)


def evaluate(segvqa, data_loader, criterion, epoch, args):
    """Calculates vqg average loss on data_loader.

    Args:
        vocab: questions and answers vocabulary.
        vqa: visual question answering model.
        data_loader: Iterator for the data.
        criterion: The criterion function used to evaluate the loss.
        args: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    segvqa.eval()
    total_loss = 0.0
    iterations = 0
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    for i, (images, questions, answers, image_masks) in enumerate(data_loader):

        # Quit after eval_steps.
        if args.eval_steps is not None and i >= args.eval_steps:
            break

        # Set mini-batch dataset.
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            image_masks = image_masks.cuda()
        alengths = process_lengths(answers)

        # Forward.
        target = image_masks[:, :, :, 0].view(-1)

        # Forward.
        logits = segvqa(images, answers, alengths=alengths, questions=questions)
        logits_flatten = flatten_logits(logits)
        loss = criterion(logits_flatten, target.long())
        # zero the parameter gradients
        logits_flatten = flatten_logits(logits)
        loss = criterion(logits_flatten, target.long())

        # Backprop and optimize.
        total_loss += loss.item()
        iterations += 1

        score, cm = get_miou(logits, target)

        # Print logs.
        if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info('Time: %.4f, Step [%d/%d], '
                             'Avg Loss: %.4f, Batch MIOU: %.4f'
                             % (delta_time, i, total_steps,
                                total_loss/iterations,
                                score))

        if iterations == 1:
            overall_confusion_matrix = cm
        else:
            overall_confusion_matrix += cm
    miou, _ = get_miou(None, None, cm)
    return miou, total_loss / iterations


def main(args):

    # Setting up seeds.
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # Create model directory.
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_dir, 'train.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Save the arguments.
    with open(os.path.join(args.model_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Image preprocessing.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)

    # Build data loader.
    logging.info("Building data loader...")
    data_loader = get_vg_loader(args.dataset, transform,
                                args.batch_size, shuffle=True,
                                num_workers=args.num_workers,
                                max_examples=args.max_examples)

    val_data_loader = get_vg_loader(args.val_dataset, transform,
                                    args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    logging.info("Done")

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
    logging.info("Done")

    if torch.cuda.is_available():
        segvqa.cuda()

    # Loss and Optimizer.
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    if torch.cuda.is_available():
        criterion.cuda()

    # Parameters to train.
    params = segvqa.params_to_train()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=True, min_lr=1e-6)

    # Train the Models.
    total_steps = len(data_loader) * args.num_epochs
    start_time = time.time()
    n_steps = 0
    for epoch in range(args.num_epochs):
        for i, (images, questions, answers, image_masks) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
                image_masks = image_masks.cuda()
            alengths = process_lengths(answers)

            targets = image_masks[:, :, :, 0].view(-1)

            # Forward.
            segvqa.train()
            segvqa.zero_grad()
            optimizer.zero_grad()
            logits = segvqa(images, answers, alengths=alengths, questions=questions)
            logits_flatten = flatten_logits(logits)
            loss = criterion(logits_flatten, targets.long())

            # Backprop and optimize.
            loss.backward()
            optimizer.step()

            # Print log info.
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                score, _ = get_miou(logits, targets)
                logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                             'MIOU: %.4f, Loss: %.4f, LR: %f'
                             % (delta_time, epoch+1, args.num_epochs, n_steps, total_steps,
                                score, loss.item(), optimizer.param_groups[0]['lr']))
                sample_image(logits, image_masks)

            # Save the models.
            if (i+1) % args.save_step == 0:
                torch.save(segvqa.state_dict(),
                           os.path.join(args.model_dir,
                                        'segvqa-%d-%d.pkl' % (epoch+1, i+1)))

        torch.save(segvqa.state_dict(), os.path.join(args.model_dir,
                   'segvqa-%d.pkl' % (epoch+1)))

        # Evaluation and learning rate updates.
        logging.info('=' * 100)
        miou, val_loss = evaluate(segvqa, val_data_loader, criterion,
                                  epoch, args)
        logging.info('Validation MIOU: %.4f' % miou)
        scheduler.step(val_loss)
        logging.info('=' * 100)

    # Save the final model.
    torch.save(segvqa.state_dict(), os.path.join(args.model_dir, 'segvqa.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Location parameters.
    parser.add_argument('--model-dir', type=str,
                        default='weights/segvqa1/',
                        help='Path for saving trained models.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/vocab_vqa_multi.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='data/vqa_seg_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/vqa_seg_val_dataset.hdf5',
                        help='path for train annotation json file')

    # Session parameters.
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for printing log info.')
    parser.add_argument('--save-step', type=int, default=10000,
                        help='Step size for saving trained models.')
    parser.add_argument('--eval-steps', type=int, default=10000,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=1000,
                        help='Run eval after every N steps.')
    parser.add_argument('--eval-all', action='store_true',
                        help='Run eval after each epoch.')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--data-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Number of data points to train with.')

    # Model parameters.
    parser.add_argument('--rnn-cell', type=str, default='lstm',
                        help='Type of rnn cell (gru or lstm).')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--vocab-embed-size', type=int, default=300,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of layers in lstm')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Whether encoder is bidirectional.')
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout.')
    parser.add_argument('--use-glove', action='store_true', default=True,
                        help='Use glove when encoding.')
    parser.add_argument('--embedding-name', type=str, default='6B',
                        help='Name of the GloVe embedding to use.')

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    Vocabulary()
