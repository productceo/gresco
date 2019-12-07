"""Add VQA multiple choice answers.
"""

from PIL import Image
from collections import Counter
from torchvision import transforms

import argparse
import json
import h5py
import nltk
import os
import progressbar

from models.EncoderCNN import SpatialResnetEncoder
from vocab import Vocabulary
from vocab import process_text


def get_top_answers(annotations):
    all_answers = []
    for d in annotations['annotations']:
        all_answers.append(d['multiple_choice_answer'])
    top_answers = [t[0] for t in Counter(all_answers).most_common(1000)]
    return top_answers


def get_qas(questions, annotations, top_answers):
    qmap = {}
    output = []
    for q in questions['questions']:
        qmap[q['question_id']] = q['question']
    for a in annotations['annotations']:
        answer = a['multiple_choice_answer']
        if answer in top_answers:
            output.append({
                'question': qmap[a['question_id']],
                'image_id': a['image_id'],
                'answer': answer,
                'category': top_answers.index(answer)})
    return output


def create_vocab(qas, threshold=4):
    counter = Counter()
    for qa in qas:
        question = qa['question'].encode('utf-8')
        answer = qa['answer'].encode('utf-8')
        qtokens = nltk.tokenize.word_tokenize(question.lower())
        atokens = nltk.tokenize.word_tokenize(answer.lower())
        counter.update(qtokens)
        counter.update(atokens)

    # If a word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Adds the words to the vocabulary.
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    return vocab


def store_qas(dataset, qas, vocab, max_length=20):
    total = len(qas)
    questions = dataset.create_dataset(
            'questions', (total, args.max_length), dtype='i')
    answers = dataset.create_dataset(
            'answers', (total, args.max_length), dtype='i')
    categories = dataset.create_dataset(
            'categories', (total,), dtype='i')
    image_indices = dataset.create_dataset(
            'image_indices', (total,), dtype='i')

    image_ids = []
    bar = progressbar.ProgressBar(maxval=len(qas))
    for idx, entry in enumerate(qas):
        i_image = len(image_ids)
        if entry['image_id'] in image_ids:
            i_image = image_ids.index(entry['image_id'])
        else:
            image_ids.append(entry['image_id'])
        image_indices[idx] = i_image
        categories[idx] = entry['category']
        q, length = process_text(entry['question'].encode('utf-8'), vocab,
                              max_length=max_length)
        questions[idx, :length] = q
        a, length = process_text(entry['answer'].encode('utf-8'), vocab,
                              max_length=max_length)
        answers[idx, :length] = a
        bar.update(idx)
    return image_ids


def store_features(dataset, image_ids, train_loc, val_loc):
    model = SpatialResnetEncoder(1, feats_available=False)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    total_images = len(image_ids)
    features = dataset.create_dataset(
            'feats', (total_images, 1024, 14, 14), dtype='f')

    bar = progressbar.ProgressBar(maxval=total_images)
    for i, image_id in enumerate(image_ids):
        path = str(image_id).zfill(12) + '.jpg'
        try:
            image = Image.open(os.path.join(train_loc, path)).convert('RGB')
        except IOError:
            image = Image.open(os.path.join(val_loc, path)).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        feature = model.resnet(image)
        features[i, :, :, :] = feature[0, :, :, :]
        bar.update(i)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum length of questions.')

    # VQA dataset
    parser.add_argument('--train-annotations', type=str,
                        default='/data/VQA/v2/v2_mscoco_train2014_annotations.json')
    parser.add_argument('--val-annotations', type=str,
                        default='/data/VQA/v2/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--train-questions', type=str,
                        default='/data/VQA/v2/v2_OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--val-questions', type=str,
                        default='/data/VQA/v2/v2_OpenEnded_mscoco_val2014_questions.json')

    # MSCOCO images.
    parser.add_argument('--train-images', type=str,
                        default='/data/MSCOCO/train2017')
    parser.add_argument('--val-images', type=str,
                        default='/data/MSCOCO/val2017')

    # HDF5 outputs.
    parser.add_argument('--top-answers', type=str,
                        default='data/processed/vqa_top_answers.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--vocab', type=str,
                        default='data/processed/vocab_vqa_multi.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--train-output', type=str,
                        default='data/processed/vqa_multi_dataset.hdf5',
                        help='Training output.')
    parser.add_argument('--val-output', type=str,
                        default='data/processed/vqa_val_multi_dataset.hdf5',
                        help='Validation output.')
    args = parser.parse_args()

    # Get the top 1000 answers.
    print('Getting top 1000 answers...')
    train_annotations = json.load(open(args.train_annotations))
    top_answers = get_top_answers(train_annotations)
    json.dump(top_answers, open(args.top_answers, 'w'))

    # Get the train qas.
    print('Parsing to get train qas...')
    train_questions = json.load(open(args.train_questions))
    train_qas = get_qas(train_questions, train_annotations, top_answers)
    print('Parsed %d train qas.' % len(train_qas))

    # Create the vocabulary.
    print('Creating vocab...')
    vocab = create_vocab(train_qas)
    vocab.save(args.vocab)

    # Store the train set in hdf5.
    print('Storing %d train qas...' % len(train_qas))
    train = h5py.File(args.train_output, 'w')
    train_image_ids = store_qas(train, train_qas, vocab, max_length=args.max_length)

    # Get the val qas.
    print('Parsing the val set...')
    val_annotations = json.load(open(args.val_annotations))
    val_questions = json.load(open(args.val_questions))
    val_qas = get_qas(val_questions, val_annotations, top_answers)

    # Store the val set in hdf5.
    print('Storing %d val set...' % len(val_qas))
    val = h5py.File(args.val_output, 'w')
    val_image_ids = store_qas(val, val_qas, vocab, max_length=args.max_length)

    # Featurize and store train image features
    print('Storing %d train features...' % len(train_image_ids))
    store_features(train, train_image_ids, args.train_images, args.val_images)

    # Featurize and store val image features
    print('Storing %d val features...' % len(val_image_ids))
    store_features(val, val_image_ids, args.train_images, args.val_images)

    # Hack so that we can import Vocabulary.
    Vocabulary()
    train.close()
    val.close()
