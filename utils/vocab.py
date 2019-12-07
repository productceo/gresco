"""Creates a vocabulary for the responses and the POS.
"""

from collections import Counter

import argparse
import json
import nltk
import logging
import numpy as np
import re
import sys

from .train_utils import Vocabulary


def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.

    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.

    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
    tokens = tokenize(text.lower().strip())
    output = []
    output.append(vocab(vocab.SYM_SOQ))  # <start>
    output.extend([vocab(token) for token in tokens])
    output.append(vocab(vocab.SYM_EOS))  # <end>
    length = min(max_length, len(output))
    if length == max_length:
        output[max_length - 1] = vocab(vocab.SYM_EOS)
    return np.array(output[:length]), length


def load_vocab(vocab_path):
    """Load Vocabulary object from a pickle file.

    Args:
        vocab_path: The location of the vocab pickle file.

    Returns:
        A Vocabulary object.
    """
    vocab = Vocabulary()
    vocab.load(vocab_path)
    return vocab


def tokenize(sentence):
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
            sentence.strip().lower().decode('utf8'))
    return tokens


def build_vocab(annotations, threshold):
    """Build a vocabulary from the annotations.

    Args:
        annotations: A json file containing the questions and responses.
        threshold: The minimum number of times a work must occur. Otherwise it
            is treated as an `Vocabulary.SYM_UNK`.

    Returns:
        A Vocabulary object.
    """
    with open(annotations) as f:
        annotations = json.load(f)

    counter = Counter()
    if args.pos:
        counter_pos = Counter()

    for i, entry in enumerate(annotations):
        question = entry["question"].encode('utf8')
        q_tokens = tokenize(question)
        counter.update(q_tokens)

        if not args.no_responses:
            response = entry["response"].encode('utf8')
            a_tokens = tokenize(response)
            counter.update(a_tokens)

        if args.pos:
            question_pos = entry["question_pos"].encode('utf8')
            qpos_tokens = tokenize(question_pos)
            counter_pos.update(qpos_tokens)
            message = '%s != %s' % ('-'.join(q_tokens), '-'.join(qpos_tokens))
            assert len(q_tokens) == len(qpos_tokens), message

            if not args.no_responses:
                response_pos = entry["response_pos"].encode('utf8')
                apos_tokens = tokenize(response_pos)
                counter_pos.update(apos_tokens)
                message = '%s != %s, %s' % ('-'.join(a_tokens),
                                            '-'.join(apos_tokens), response)
                assert len(a_tokens) == len(apos_tokens), message

        if i % 1000 == 0:
            logging.info("Tokenized %d questions." % (i))

    # If a word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = create_vocab(words)

    vocab_pos = None
    if args.pos:
        words_pos = [word for word, cnt in counter_pos.items()
                     if cnt >= threshold]
        vocab_pos = create_vocab(words_pos)

    return vocab, vocab_pos


def create_vocab(words):
    # Adds the words to the vocabulary.
    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-path', type=str,
                        default='/scr/junwon/gresco/datasets/input/vocab_vqa_multi.json',
                        help='Path for saving vocabulary wrapper.')
    parser.add_argument('--annotations', type=str,
                        default='data/processed/respeval_train_pos.json',
                        help='Path for train annotation file.')
    parser.add_argument('--threshold', type=int, default=4,
                        help='Minimum word count threshold.')
    parser.add_argument('--pos', action='store_true', default=False,
                        help='Use POS features.')
    parser.add_argument('--vocab-pos-path', type=str,
                        default='data/processed/vocab_pos.pkl',
                        help='Path for saving pos vocabulary wrapper.')
    parser.add_argument('--no-responses', action='store_true', default=False,
                        help='Set to true if annotations does not have '
                             'responses.')
                        
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    vocab, vocab_pos = build_vocab(args.annotations, args.threshold)
    logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)
    if args.pos:
        vocab_pos.save(args.vocab_pos_path)
        logging.info("Saved the POS vocabulary wrapper to '%s'"
                     % args.vocab_pos_path)