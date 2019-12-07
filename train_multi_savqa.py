"""Training script to train a MultiSAVQA classification model.
"""

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import json
import logging
import time
import torch
import os

from utils import accuracy
from utils import get_vqa_loader
from models import MultiSAVQAModel
from utils import compare_outputs
from utils import Vocabulary
from utils import load_vocab
from utils import process_lengths


def evaluate(vocab, vqa, data_loader, criterion, epoch, args):
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
    gts, gens, qs = [], [], []
    vqa.eval()
    total_loss = 0.0
    total_correct = 0.0
    iterations = 0
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    for i, (feats, questions, categories) in enumerate(data_loader):

        # Set mini-batch dataset.
        if torch.cuda.is_available():
            feats = feats.cuda()
            questions = questions.cuda()
            categories = categories.cuda()
        qlengths = process_lengths(questions)

        # Forward.
        outputs = vqa(feats, questions, qlengths)
        loss = criterion(outputs, categories)
        preds = outputs.max(1)[1]

        # Backprop and optimize.
        total_loss += loss.item()
        total_correct += accuracy(preds, categories)
        iterations += 1

        # Quit after eval_steps.
        if args.eval_steps is not None and i >= args.eval_steps:
            break
        q, gen, gt = parse_outputs(preds, questions, categories, vocab)
        gts.extend(gt)
        gens.extend(gen)
        qs.extend(q)

        # Print logs.
        if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info(
                        'Time: %.4f, Step [%d/%d], '
                        'Avg Loss: %.4f, Avg Acc: %.4f'
                             % (delta_time, i, total_steps,
                                total_loss/iterations,
                                total_correct/iterations))
    # Compare model reconstruction to target
    compare_outputs(gens, qs, gts, logging)
    return total_loss / iterations


def parse_outputs(preds, questions, categories, vocab):
    """Converts the model's outputs to actual words.

    Args:
        preds: The max category predicted by the model.
        questions: The indices of the questions.
        categories: Ground truth answer categories.
        vocab: The Vocabulary instance used by the model.

    Returns:
        qs: List of strings of questions.
        generated: List of predicted answers.
        ytrue: List of string of gt answers.
    """
    qs, generated, ytrue = [], [], []
    for i in range(categories.size(0)):
        question = vocab.tokens_to_words(questions[i])
        output = vocab.top_answers[preds[i].item()]
        category = vocab.top_answers[categories[i].item()]
        ytrue.append(str(category))
        qs.append(str(question))
        generated.append(str(output))
    return qs, generated, ytrue


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

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)
    vocab.top_answers = json.load(open(args.top_answers))

    # Build data loader.
    logging.info("Building data loader...")
    data_loader = get_vqa_loader(args.dataset, args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples)

    val_data_loader = get_vqa_loader(args.val_dataset, args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)
    logging.info("Done")

    # Build the models
    logging.info("Building MultiSAVQA models...")
    vqa = MultiSAVQAModel(
            len(vocab),
            args.max_length,
            args.hidden_size,
            args.vocab_embed_size,
            num_layers=args.num_layers,
            rnn_cell=args.rnn_cell,
            bidirectional=args.bidirectional,
            input_dropout_p=args.dropout,
            dropout_p=args.dropout,
            num_att_layers=args.num_att_layers,
            att_ff_size=args.att_ff_size)
    logging.info("Done")

    if torch.cuda.is_available():
        vqa.cuda()

    # Loss and Optimizer.
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()

    # Parameters to train.
    params = vqa.params_to_train()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=True, min_lr=1e-6)

    # Train the Models.
    total_steps = len(data_loader) * args.num_epochs
    start_time = time.time()
    n_steps = 0
    for epoch in range(args.num_epochs):
        for i, (feats, questions, categories) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                feats = feats.cuda()
                questions = questions.cuda()
                categories = categories.cuda()
            qlengths = process_lengths(questions)

            # Forward.
            vqa.train()
            vqa.zero_grad()
            outputs = vqa(feats, questions, qlengths)

            # Calculate the loss.
            loss = criterion(outputs, categories)

            # Backprop and optimize.
            loss.backward()
            optimizer.step()

            # Eval now.
            if (args.eval_every_n_steps is not None and
                    n_steps >= args.eval_every_n_steps and
                    n_steps % args.eval_every_n_steps == 0):
                logging.info('=' * 100)
                val_loss = evaluate(vocab, vqa, val_data_loader, criterion,
                         epoch, args)
                scheduler.step(val_loss)
                logging.info('=' * 100)

            # Take argmax for each timestep
            preds = outputs.max(1)[1]
            score = accuracy(preds, categories)

            # Print log info.
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                             'Accuracy: %.4f, Loss: %.4f, LR: %f'
                      % (delta_time, epoch+1, args.num_epochs, n_steps, total_steps,
                         score, loss.item(), optimizer.param_groups[0]['lr']))

            # Save the models.
            if (i+1) % args.save_step == 0:
                torch.save(vqa.state_dict(),
                           os.path.join(args.model_dir,
                                        'multi-savqa-%d-%d.pkl' %(epoch+1, i+1)))

        torch.save(vqa.state_dict(), os.path.join(args.model_dir,
                   'multi-savqa-%d.pkl' % (epoch+1)))

        # Evaluation and learning rate updates.
        logging.info('=' * 100)
        val_loss = evaluate(vocab, vqa, val_data_loader, criterion,
                            epoch, args)
        scheduler.step(val_loss)
        logging.info('=' * 100)

    # Save the final model.
    torch.save(vqa.state_dict(),os.path.join(args.model_dir,'vqa.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Location parameters.
    parser.add_argument('--model-dir', type=str,
                        default='weights/savqa1/',
                        help='Path for saving trained models.')
    parser.add_argument('--vocab-path', type=str,
                        default='/scr/junwon/gresco/datasets/input/vocab_vqa_multi.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='data/processed/vqa_multi_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/processed/vqa_val_multi_dataset.hdf5',
                        help='path for train annotation json file')
    parser.add_argument('--top-answers', type=str,
                        default='data/processed/vqa_top_answers.json',
                        help='Path for vocabulary wrapper.')

    # Session parameters.
    parser.add_argument('--log-step', type=int , default=10,
                        help='Step size for printing log info.')
    parser.add_argument('--save-step', type=int , default=10000,
                        help='Step size for saving trained models.')
    parser.add_argument('--eval-steps', type=int, default=None,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=None,
                        help='Run eval after every N steps.')
    parser.add_argument('--eval-all', action='store_true',
                        help='Run eval after each epoch.')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=4e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-examples', type=int , default=None,
                        help='Number of data points to train with.')

    # Model parameters.
    parser.add_argument('--rnn-cell', type=str, default='LSTM',
                        help='Type of rnn cell (gru or lstm).')
    parser.add_argument('--hidden-size', type=int , default=1024 ,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--vocab-embed-size', type=int , default=500,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-layers', type=int , default=2,
                        help='number of layers in lstm')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Whether encoder is bidirectional.')
    parser.add_argument('--max-length', type=int , default=20,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float , default=0.5,
                        help='Dropout.')

    # SAVQA-specific parameters.
    parser.add_argument('--num-att-layers', type=int , default=2,
                        help='Number of stacked attention layers to use.')
    parser.add_argument('--att-ff-size', type=int , default=512,
                        help='Dimension size of stacked attention.')

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    Vocabulary()
