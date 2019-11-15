"""Defines the Vocabulary object and builds it from a dataset.
"""

from collections import Counter
from train_utils import Vocabulary

import argparse
import json
import nltk


def build_vocab(qa_json, threshold, include_answers=True):
    """Build a simple vocabulary wrapper.
    """
    with open(qa_json) as f:
        vg_qas = json.load(f)

    num_questions = 0
    counter = Counter()
    for entry in vg_qas:
        for qa in entry["qas"]:
            question = qa["question"].encode('utf-8')
            tokens = nltk.tokenize.word_tokenize(question.lower())
            counter.update(tokens)

            if include_answers:
                answer = qa["answer"].encode('utf-8')
                tokens = nltk.tokenize.word_tokenize(answer.lower())
                counter.update(tokens)

            num_questions += 1

            if num_questions % 1000 == 0:
                print("Tokenized %d questions." % num_questions)

    # If a word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Adds the words to the vocabulary.
    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(qa_json=args.annotations,
                        threshold=args.threshold,
                        include_answers=not(args.without_answers))
    vocab.save(args.vocab_path)
    print("Total vocabulary size: %d" % len(vocab))
    print("Saved the vocabulary wrapper to '%s'" % args.vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str,
                        default='data/filtered_qas_train_neutral.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_vg.json',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--without-answers', action='store_true',
                        help='do not include answers in the vocab')
    args = parser.parse_args()
    main(args)
