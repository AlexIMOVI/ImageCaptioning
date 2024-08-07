from DenseCap.densecap.densecap_utils import utils
from DenseCap.densecap import box_utils
import torch
import torchvision
import subprocess
from easydict import EasyDict as edict
from nltk.translate import meteor
from nltk import word_tokenize


def pluck_boxes(ix, boxes, text):
    N = len(ix)
    new_boxes = torch.zeros(N, 4, device='cpu')
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

class DenseCaptioningEvaluator:
    def __init__(self, opt):
        self.all_logprobs = []
        self.records = []
        self.n = 1
        self.npos = 0
        self.id = utils.getopt(opt, 'id', '')
        self.device = 'cpu'

    def addResult(self, logprobs, boxes, text, target_boxes, target_text):
        assert logprobs.size(0) == boxes.size(0)
        assert logprobs.size(0) == len(text)
        assert target_boxes.size(0) == len(target_text)
        assert boxes.dim() == 2

        boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
        target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes)

        boxes = boxes.float().to(self.device)
        logprobs = logprobs.double().to(self.device)
        target_boxes = target_boxes.float().to(self.device)

        mergeix = box_utils.merge_boxes(target_boxes, .7)
        merged_boxes, merged_text = pluck_boxes(mergeix, target_boxes, target_text)

        Y, IX = torch.sort(logprobs, 0, True)

        nd = logprobs.size(0)
        nt = merged_boxes.size(0)
        used = torch.zeros(nt, device=self.device)
        ov_matrix = torchvision.ops.box_iou(merged_boxes, boxes)

        for d in range(nd):
            ii = IX[d]
            ovmax = torch.zeros(1, device=self.device)
            jmax = 0
            j_ok = False
            for j in range(nt):
                ov = ov_matrix[j, ii]
                if ov > ovmax:
                    ovmax = ov
                    jmax = j
                    j_ok = True
            ok = 1
            if used[jmax] == 0:
                used[jmax] = 1  # mark as taken
            else:
                ok = 0

            # record the best box, the overlap, and the fact that we need to score the language match
            record = edict()
            record.ok = ok  # whether this prediction can be counted toward a true positive
            record.ov = ovmax.item()
            record.candidate = text[ii]
            record.references = merged_text[jmax] if j_ok else []
            record.imgid = self.n
            self.records.append(record)

        # keep track of results
        self.n += 1
        self.npos += nt
        self.all_logprobs.append(Y.double())  # inserting the sorted logprobs as double

    def evaluate(self, verbose=None):
        if verbose is None:
            verbose = True
        min_overlaps = [.3, .4, .5, .6, .7]
        min_scores = [-1, 0, .05, .1, .15, .2, .25]

        logprobs = torch.cat(self.all_logprobs)
        blob = score_captions(self.records)
        scores = blob['scores']

        if verbose:
            for k in range(len(self.records)):
                record = self.records[k]
                if record['ov'] > 0 and k % 1000 == 0:
                    txtgt = ''
                    assert type(record.references) == list

                    for vv in record.references:
                        txtgt = vv + '. '
                    print(f'IMG {record.imgid}, PRED: {record.candidate},'
                          f' GT: {txtgt}, OK: {record.ok}, OV: {record.ov}, SCORE: {scores[k]}')

        y, ix = torch.sort(logprobs, 0, True)

        ap_results = {}
        det_results = {}

        for min_overlap in min_overlaps:
            for min_score in min_scores:
                n = y.numel()
                tp = torch.zeros(n, device=self.device)
                fp = torch.zeros(n, device=self.device)

                for i in range(n):
                    ii = ix[i]
                    r = self.records[ii]

                    if len(r.references) < 0:
                        fp[i] = 1
                    else:
                        score = scores[ii]
                        if r.ov >= min_overlap and r.ok == 1 and score > min_score:
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

                if min_score == -1:
                    det_results['ov'+str(min_overlap)] = ap.item()
                else:
                    ap_results['ov'+str(min_overlap)+'score'+str(min_score)] = ap.item()

        map = utils.average_values(ap_results)
        detmap = utils.average_values(det_results)

        results = {'map': map, 'ap_breakdown': ap_results,
                   'detmap': detmap, 'det_breakdown': det_results
                   }
        return results

    def numAdded(self):
        return self.n - 1

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
    all_losses = []
    while True:
        counter += 1

        # Grab a batch of data and convert it to the right dtype
        loader_kwargs = {'split': split, 'iterate': True}
        data = edict()
        data.image, data.gt_boxes, data.gt_labels, info, _ = loader.get_batch(loader_kwargs)
        info = info[0]  # Since we are only using a single image

        # Call forward_backward to compute losses
        model.timing = False
        model.dump_vars = False
        model.cnn_backward = False
        model.nets['localization_layer'].eval_mode = True
        losses = model.forward_backward(data)
        all_losses.append(losses)

        # Call forward_test to make predictions, and pass them to evaluator
        model.nets['localization_layer'].eval_mode = False
        torch.cuda.synchronize()
        boxes, logprobs, captions = model.forward_test(data.image)
        torch.cuda.synchronize()
        gt_captions = model.nets['llm'].decode_sequence(data.gt_labels[0])
        evaluator.addResult(logprobs, boxes, captions, data.gt_boxes[0], gt_captions)

        # Print a message to the console
        msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
        num_images = info['split_bounds'][1]
        if max_images > 0:
            num_images = min(num_images, max_images)
        num_boxes = boxes.size(0)
        print(msg % (info['filename'], counter, num_images, split, num_boxes))

        # Break out if we have processed enough images
        if max_images > 0 and counter >= max_images:
            break
        if info['split_bounds'][0] == info['split_bounds'][1]:
            break

    loss_results = utils.dict_average(all_losses)
    # print('Loss stats:')
    # print(loss_results)
    # print('Average loss: ', loss_results['total_loss'])

    ap_results = evaluator.evaluate(verbose=True)
    print('mAP: %f' % (100 * ap_results['map']))

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
    # utils.write_json('eval/input.json', records)
    # # subprocess.Popen([sys.executable])
    # subprocess.run('python3 eval/meteor_bridge.py', shell=True)
    # blob = utils.read_json('eval/output.json')
    return blob

