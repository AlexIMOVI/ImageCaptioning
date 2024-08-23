from AlexCap.densecap_utils import utils
import torch
from easydict import EasyDict as edict
from nltk.translate import meteor, bleu_score
from nltk import word_tokenize

class DenseCaptioningEvaluator:
    def __init__(self, opt):
        self.records = []
        self.n = 1
        self.npos = 0
        self.device = 'cpu'

    def addResult(self, text, target_text, info):

        for i, t in enumerate(text):

            record = edict()
            record.candidate = t
            record.references = target_text[i]
            record.imgid = info['filename'][i]
            self.records.append(record)

        # keep track of results
        self.n += 1
        self.npos += 1

    def evaluate(self, verbose=None):
        if verbose is None:
            verbose = True
        blob = score_captions(self.records)
        scores = blob['scores']
        bl_scores = blob['bleu_scores']
        if verbose:
            for k in range(len(self.records)):
                record = self.records[k]
                if k % 10 == 0:
                    print(f'IMG {record.imgid}, PRED: {record.candidate},'
                          f' GT: {record.references}, SCORE: M>{scores[k]}, BLEU>{bl_scores[k]}')
        results = {'meteor': blob['average_score'], 'bleu': blob['average_bl_score']}
        return results

def eval_split(kwargs):
    model = utils.getopt(kwargs, 'model')
    loader = utils.getopt(kwargs, 'loader')
    split = utils.getopt(kwargs, 'split', 'val')
    max_images = utils.getopt(kwargs, 'max_images', -1)
    batch_size = utils.getopt(kwargs, 'val_batch_size', 1)

    assert split == 'val' or split == 'test', 'split must be "val" or "test"'
    split_to_int = {'val': 1, 'test': 2}
    split = split_to_int[split]
    print('using split ', split)

    model.eval()
    loader.reset_iterator(split)
    evaluator = DenseCaptioningEvaluator(id)
    counter = 0
    all_losses = 0
    while True:
        counter += batch_size

        # Grab a batch of data and convert it to the right dtype
        loader_kwargs = {'split': split, 'iterate': True}
        data = edict()
        data.image, data.gt_labels, info, _ = loader.get_batch(loader_kwargs, batch_size)
        info = info[0]  # Since we are only using a single image

        # Call forward_backward to compute losses
        model.timing = False
        model.dump_vars = False
        model.cnn_backward = False
        model.set_eval(True)
        losses = model.forward_train(data)
        all_losses += losses.item()

        # Call forward_test to make predictions, and pass them to evaluator
        model.set_eval(False)
        captions = model.forward_test(data)
        captions = captions[0] if isinstance(captions, tuple) else captions
        gt_captions = model.llm.decode_sequence(data.gt_labels)
        evaluator.addResult(captions, gt_captions, info)

        # Print a message to the console
        msg = 'Processed image %s (%d / %d) of split %d'
        num_images = info['split_bounds'][1]
        if max_images > 0:
            num_images = min(num_images, max_images)
            counter = min(counter, max_images)
        counter = min(counter, num_images)
        print(msg % (info['filename'][0]+'...', counter, num_images, split))

        # Break out if we have processed enough images
        if max_images > 0 and counter >= max_images:
            break
        if counter >= num_images:
            break

    loss_results = batch_size*all_losses / counter
    # print('Loss stats:')
    # print(loss_results)
    # print('Average loss: ', loss_results['total_loss'])

    ap_results = evaluator.evaluate(verbose=True)
    print(f'METEOR: {ap_results["meteor"]}')
    print(f'BLEU: {ap_results["bleu"]}')
    out = {
        'loss_results': loss_results,
        'ap_results': ap_results,
    }
    return out


def score_captions(records):
    smooth = bleu_score.SmoothingFunction().method4
    blob = {}
    scores = []
    bl_scores = []
    for r in records:
        bl_score = round(bleu_score.sentence_bleu([word_tokenize(r['references'])], word_tokenize(r['candidate']), smoothing_function=smooth), 4)
        bl_scores.append(bl_score)
        score = round(meteor([word_tokenize(r['references'])], word_tokenize(r['candidate'])), 4)
        scores.append(score)
    blob['scores'] = scores
    blob['bleu_scores'] = bl_scores
    blob['average_bl_score'] = sum(bl_scores) / len(bl_scores)
    blob['average_score'] = sum(scores) / len(scores)

    return blob

