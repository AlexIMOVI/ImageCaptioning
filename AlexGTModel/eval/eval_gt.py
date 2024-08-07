from AlexGTModel.densecap_utils import utils
import torch
import subprocess
from easydict import EasyDict as edict
import torchvision
from nltk.translate import meteor
from nltk import word_tokenize
class DenseCaptioningEvaluator:
    def __init__(self, opt):
        self.records = []
        self.n = 1
        self.npos = 0
        self.id = utils.getopt(opt, 'id', '')
        self.device = 'cpu'

    def pluck_boxes(self, ix, boxes, text):
        N = len(ix)
        new_boxes = torch.zeros(N, 4, device=self.device)
        new_text = []

        for i in range(N):
            ixi = ix[i]
            n = ixi.numel()
            bsub = boxes.index_select(0, ixi)
            newbox = torch.mean(bsub, 0)
            new_boxes[i] = newbox

            texts = []
            if len(text) > 0:
                for j in range(n):
                    texts.append(text[ixi[j]])

            new_text.append(texts)

        return new_boxes, new_text

    def xcycwh_to_x1y1x2y2(self, boxes):
        minibatch = True
        if boxes.dim() == 2:
            minibatch = False
            boxes = boxes.unsqueeze(0)

        ret = torch.zeros_like(boxes, device=self.device)

        xc = boxes[:, :, 0]
        yc = boxes[:, :, 1]
        w = boxes[:, :, 2]
        h = boxes[:, :, 3]

        ret[:, :, 0] = (1 - w) / 2 + xc
        ret[:, :, 1] = (1 - h) / 2 + yc
        ret[:, :, 2] = (w - 1) / 2 + xc
        ret[:, :, 3] = (h - 1) / 2 + yc

        if not minibatch:
            ret = ret.squeeze(0)

        return ret

    def merge_boxes(self, boxes, thr):
        assert thr > 0
        ix = []
        D = torchvision.ops.box_iou(boxes, boxes)
        while True:
            good = torch.ge(D, .7)
            good_sum = torch.sum(good, 0).view(-1)
            topnum, topix = torch.max(good_sum, dim=0)
            if topnum.item() == 0:
                break
            mergeix = torch.nonzero(good[topix]).view(-1)

            ix.append(mergeix)
            D.index_fill_(0, mergeix, 0)
            D.index_fill_(1, mergeix, 0)

        return ix

    def addResult(self, boxes, text, target_text, info):

        boxes = self.xcycwh_to_x1y1x2y2(boxes)
        boxes = boxes.float().to(self.device)
        mergeix = self.merge_boxes(boxes, .7)
        merged_boxes, merged_text = self.pluck_boxes(mergeix, boxes, target_text)
        nd = boxes.size(0)
        nt = merged_boxes.size(0)
        ov_matrix = torchvision.ops.box_iou(merged_boxes, boxes)
        used = torch.zeros(nt, device=self.device)
        for i in range(nd):
            ovmax = torch.zeros(1, device=self.device)
            jmax = 0
            for j in range(nt):
                ov = ov_matrix[j, i]
                if ov > ovmax:
                    ovmax = ov
                    jmax = j
            ok = 1
            if used[jmax] == 0:
                used[jmax] = 1  # mark as taken
            else:
                ok = 0
            # record the best box, the overlap, and the fact that we need to score the language match
            record = edict()
            record.ok = ok
            record.candidate = text[i]
            record.references = merged_text[jmax] #if isinstance(merged_text[jmax], list) else [merged_text[jmax]]
            record.imgid = info['filename']
            self.records.append(record)

        # keep track of results
        self.n += 1
        self.npos += nt

    def evaluate(self, verbose=None):
        if verbose is None:
            verbose = True

        min_scores = [0, .05, .1, .15, .2, .25]
        blob = score_captions(self.records)
        scores = blob['scores']

        if verbose:
            for k in range(len(self.records)):
                record = self.records[k]
                if k % 1000 == 0:
                    txtgt = ''
                    assert type(record.references) == list

                    for vv in record.references:
                        txtgt += vv + '. '
                    print(f'IMG {record.imgid}, PRED: {record.candidate},'
                          f' GT: {txtgt}, SCORE: {scores[k]}')

        ap_results = {}
        for min_score in min_scores:
            n = len(scores)
            tp = torch.zeros(n, device=self.device)
            fp = torch.zeros(n, device=self.device)

            for i in range(n):
                    score = scores[i]
                    r = self.records[i]
                    if score > min_score and r.ok == 1:
                        tp[i] = 1
                    else:
                        fp[i] = 1

            fp = torch.cumsum(fp,0)
            tp = torch.cumsum(tp, 0)
            rec = torch.div(tp, self.npos)
            prec = torch.div(tp, fp + tp)

            ap = torch.zeros(1, device=self.device)
            apn = 0

            for t in range(101):
                t = t/100.0
                mask = torch.ge(rec, t).double()
                prec_masked = torch.mul(prec.double(), mask)
                p = torch.max(prec_masked)
                ap += p
                apn += 1

            ap = ap/apn
            ap_results['score'+str(min_score)] = ap.item()

        map = utils.average_values(ap_results)
        results = {'map': map, 'ap_breakdown': ap_results, 'meteor': blob['average_score']}
        return results


def eval_split(kwargs):
    model = utils.getopt(kwargs, 'model')
    loader = utils.getopt(kwargs, 'loader')
    split = utils.getopt(kwargs, 'split', 'val')
    max_images = utils.getopt(kwargs, 'max_images', -1)
    id = utils.getopt(kwargs, 'id', '')

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
        counter += 1

        # Grab a batch of data and convert it to the right dtype
        loader_kwargs = {'split': split, 'iterate': True}
        data = edict()
        data.image, data.gt_boxes, data.gt_labels, info = loader.get_batch(loader_kwargs)
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
        torch.cuda.synchronize()
        captions = model.forward_test(data)
        torch.cuda.synchronize()
        gt_captions = model.llm.decode_sequence(data.gt_labels[0])
        evaluator.addResult(data.gt_boxes[0], captions, gt_captions, info)

        # Print a message to the console
        msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
        num_images = info['split_bounds'][1]
        if max_images > 0:
            num_images = min(num_images, max_images)
        num_labels = len(captions)
        print(msg % (info['filename'], counter, num_images, split, num_labels))

        # Break out if we have processed enough images
        if counter >= max_images > 0:
            break
        if info['split_bounds'][0] == info['split_bounds'][1]:
            break

    loss_results = all_losses / counter
    # print('Loss stats:')
    # print(loss_results)
    # print('Average loss: ', loss_results['total_loss'])

    ap_results = evaluator.evaluate(verbose=True)
    print('mAP: %f' % (100 * ap_results['map']))
    print(f'METEOR: {ap_results["meteor"]}')
    out = {
        'loss_results': loss_results,
        'ap_results': ap_results,
    }
    return out


def score_captions(records):
    blob = {}
    scores = []
    for r in records:
        score = round(meteor([word_tokenize(a) for a in r['references']], word_tokenize(r['candidate'])), 4)
        scores.append(score)
    blob['scores'] = scores
    blob['average_score'] = sum(scores) / len(scores)

    return blob
