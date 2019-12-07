"""This script is used to evaluate a question generation model.
"""

import argparse
import json
import logging
import os
import progressbar
import torch

from models import MultiSAVQAModel
from utils import get_vqa_loader
from utils import Dict2Obj
from utils import Vocabulary
from utils import load_vocab
from utils import process_lengths
from train_multi_savqa import parse_outputs

from sklearn.metrics import accuracy_score

def evaluate(vqa, data_loader, vocab, args, params):
    """Runs BLEU, METEOR, CIDEr and distinct n-gram scores.

    Args:
        vqa: question answering model.
        data_loader: Iterator for the data.
        args: ArgumentParser object.
        params: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    vqa.eval()
    preds = []
    gts = []
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    bar = progressbar.ProgressBar(maxval=total_steps)
    for iterations, (images, questions, categories) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            categories = categories.cuda()
        qlengths = process_lengths(questions)

        # Predict.
        outputs = vqa(images, questions, qlengths)
        out = outputs.max(1)[1]

        _, pred, gt = parse_outputs(out, questions, categories, vocab)
        gts.extend(gt)
        preds.extend(pred)

        bar.update(iterations)
        if args.eval_steps is not None and iterations >= args.eval_steps:
            break

    print ('='*80)
    print ('GROUND TRUTH')
    print (gts[:args.num_show])
    print ('-'*80)
    print ('PREDICTIONS')
    print (preds[:args.num_show])
    print ('='*80)
    scores = accuracy_score(gts, preds)
    return scores, gts, preds


def main(args):
    """Loads the model and then calls evaluate().

    Args:
        args: Instance of ArgumentParser.
    """

    # Load the arguments.
    model_dir = os.path.dirname(args.model_path)
    params = Dict2Obj(json.load(
            open(os.path.join(model_dir, "args.json"), "r")))

    # Config logging
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(model_dir, 'eval.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Load vocabulary wrapper.
    vocab = load_vocab(params.vocab_path)
    vocab.top_answers = json.load(open(args.top_answers))

    # Build data loader.
    logging.info("Building data loader...")
    data_loader = get_vqa_loader(args.dataset,
                                 args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples)
    logging.info("Done")

    # Build the models
    logging.info("Loading model.")
    vqa = MultiSAVQAModel(len(vocab), params.max_length, params.hidden_size,
                          params.vocab_embed_size,
                          num_layers=params.num_layers,
                          rnn_cell=params.rnn_cell,
                          bidirectional=params.bidirectional,
                          num_att_layers=params.num_att_layers,
                          att_ff_size=params.att_ff_size)
    vqa.load_state_dict(torch.load(args.model_path))
    vqa.eval()
    logging.info("Done")

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        vqa.cuda()

    scores, gts, preds = evaluate(vqa, data_loader, vocab, args, params)

    # Print and save the scores.
    print ("Accuracy Score: {}".format(scores))
    with open(os.path.join(model_dir, args.results_path), 'w') as results_file:
        json.dump(scores, results_file)
    with open(os.path.join(model_dir, args.preds_path), 'w') as preds_file:
        json.dump(preds, preds_file)
    with open(os.path.join(model_dir, args.gts_path), 'w') as gts_file:
        json.dump(gts, gts_file)
    with open(os.path.join(model_dir, args.preds_gts_path), 'w') as preds_gts_file:
        json.dump({'preds': preds, 'gts': gts}, preds_gts_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='weights/multi-savqa-v2-1.0/multi-savqa-10.pkl',
                        help='Path for loading trained models')
    parser.add_argument('--results-path', type=str, default='results.json',
                        help='Path for saving results.')
    parser.add_argument('--preds-path', type=str, default='preds.json',
                        help='Path for saving predictions.')
    parser.add_argument('--gts-path', type=str, default='gts.json',
                        help='Path for saving ground truth.')
    parser.add_argument('--preds-gts-path', type=str, default='preds_gts.json',
                        help='Path for saving ground truth.')
    parser.add_argument('--eval-steps', type=int, default=None,
                        help='Number of eval steps to run.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='When set, only evaluates that many data points.')
    parser.add_argument('--num-show', type=int, default=10,
                        help='Number of predictions to print.')

    # Data parameters.
    parser.add_argument('--dataset', type=str,
                        default='/scr/junwon/gresco/datasets/release/val_g.hdf5',
                        help='path for val hdf5 file')
    parser.add_argument('--top-answers', type=str,
                        default='/scr/junwon/gresco/datasets/input/vqa_top_answers.json',
                        help='Path for top answers.')

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    Vocabulary()
