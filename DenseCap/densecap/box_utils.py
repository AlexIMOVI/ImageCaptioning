import torch
import torch.nn as nn
from DenseCap.train_opts import get_config
import torchvision
opt = get_config()

def xcycwh_to_x1y1x2y2(boxes):
    minibatch = True
    if boxes.dim() == 2:
        minibatch = False
        boxes = boxes.unsqueeze(0)

    ret = torch.zeros_like(boxes, device=opt['device'])

    xc = boxes[:, :, 0]
    yc = boxes[:, :, 1]
    w = boxes[:, :, 2]
    h = boxes[:, :, 3]

    ret[:, :, 0] = (1-w)/2 + xc
    ret[:, :, 1] = (1-h)/2 + yc
    ret[:, :, 2] = (w-1)/2 + xc
    ret[:, :, 3] = (h-1)/2 + yc

    if not minibatch:
        ret = ret.squeeze(0)

    return ret

def xywh_to_x1y1x2y2(boxes):
    minibatch = True
    if boxes.dim() == 2:
        minibatch = False
        boxes = boxes.unsqueeze(0)

    x = boxes[:, :, 0]
    y = boxes[:, :, 1]
    w = boxes[:, :, 2]
    h = boxes[:, :, 3]

    ret = torch.zeros_like(boxes, device=opt['device'])
    ret[:, :, 0] = x
    ret[:, :, 1] = y
    ret[:, :, 2] = x + w - 1
    ret[:, :, 3] = y + h - 1

    if not minibatch:
        ret = ret.squeeze(0)

    return ret

def x1y1x2y2_to_xywh(boxes):
    minibatch = True
    if boxes.dim() == 2:
        minibatch = False
        boxes = boxes.unsqueeze(0)

    x0 = boxes[:, :, 0]
    y0 = boxes[:, :, 1]
    x1 = boxes[:, :, 2]
    y1 = boxes[:, :, 3]
    ret = torch.zeros_like(boxes, device=opt['device'])
    ret[:, :, 0] = x0
    ret[:, :, 1] = y0
    ret[:, :, 2] = x1 - x0 + 1
    ret[:, :, 3] = y1 - y0 + 1

    # ret = torch.stack([x, y, w, h], dim=-1)

    if not minibatch:
        ret = ret.squeeze(0)

    return ret

def x1y1x2y2_to_xcycwh(boxes):
    minibatch = True
    if boxes.dim() == 2:
        minibatch = False
        boxes = boxes.unsqueeze(0)

    x0 = boxes[:, :, 0]
    y0 = boxes[:, :, 1]
    x1 = boxes[:, :, 2]
    y1 = boxes[:, :, 3]
    ret = torch.zeros_like(boxes, device=opt['device'])
    ret[:, :, 0] = (x0 + x1)/2.0
    ret[:, :, 1] = (y0 + y1)/2.0
    ret[:, :, 2] = x1 - x0 + 1
    ret[:, :, 3] = y1 - y0 + 1

    if not minibatch:
        ret = ret.squeeze(0)

    return ret

def xywh_to_xcycwh(boxes):
    minibatch = True
    if boxes.dim() == 2:
        minibatch = False
        boxes = boxes.unsqueeze(0)

    x0 = boxes[:, :, 0]
    y0 = boxes[:, :, 1]
    w0 = boxes[:, :, 2]
    h0 = boxes[:, :, 3]
    ret = torch.zeros_like(boxes, device=opt['device'])
    ret[:, :, 0] = (w0/2 + x0)
    ret[:, :, 1] = (h0/2 + y0)
    ret[:, :, 2] = w0
    ret[:, :, 3] = h0

    # ret = torch.stack([xc, yc, w, h], dim=-1)

    if not minibatch:
        ret = ret.squeeze(0)

    return ret

def xcycwh_to_xywh(boxes):
    boxes_x1x2y1y2 = xcycwh_to_x1y1x2y2(boxes)
    boxes_xywh = x1y1x2y2_to_xywh(boxes_x1x2y1y2)

    return boxes_xywh

def scale_box_xywh(boxes, frac):
    boxes_scaled = boxes.clone()
    boxes_scaled[:, :2].add(-1)
    boxes_scaled.mul(frac)
    boxes_scaled[:, :2].add(1)
    return boxes_scaled

def clip_boxes(boxes, bounds, format):
    if (format == 'x1y1x2y2'):
        boxes_clipped = boxes.clone()
    elif (format == 'xcycwh'):
        boxes_clipped = xcycwh_to_x1y1x2y2(boxes)
    elif (format == 'xywh'):
        boxes_clipped = xywh_to_x1y1x2y2(boxes)
    else:
        raise ValueError(f'Unrecognised box format "{format}"')

    if boxes_clipped.dim() == 3:
        boxes_clipped = boxes_clipped.view(-1,4)

    boxes_clipped[:, 0] = boxes_clipped[:, 0].clamp_(bounds.x_min, bounds.x_max - 1)
    boxes_clipped[:, 1] = boxes_clipped[:, 1].clamp_(bounds.y_min, bounds.y_max - 1)
    boxes_clipped[:, 2] = boxes_clipped[:, 2].clamp_(bounds.x_min + 1, bounds.x_max)
    boxes_clipped[:, 3] = boxes_clipped[:, 3].clamp_(bounds.y_min + 1, bounds.y_max)

    validx = torch.gt(boxes_clipped[:, 2], boxes_clipped[:, 0]).byte()
    validy = torch.gt(boxes_clipped[:, 3], boxes_clipped[:, 1]).byte()
    valid = torch.gt(validx*validy, 0)

    if format == 'xcycwh':
        boxes_clipped = x1y1x2y2_to_xcycwh(boxes_clipped)
    elif format == 'xywh':
        boxes_clipped = x1y1x2y2_to_xywh(boxes_clipped)

    return boxes_clipped.view_as(boxes), valid


def eval_box_recalls(boxes, gt_boxes, ns):
    iou_threshs = [.5, .7, .9]
    if ns == None:
        ns = [100, 200, 300]

    stats = []
    from BoxIoU import BoxIoU
    boxes_view = boxes.view(1, -1, 4)
    gt_boxes_view = gt_boxes.view(1, -1, 4)
    box_iou = BoxIoU().type(boxes.type())
    ious = box_iou.forward([boxes_view,gt_boxes_view])
    ious = ious[0]
    for thresh in iou_threshs:
        mask = torch.gt(ious, thresh)
        hit = torch.gt(torch.cumsum(mask, 0), 0)
        recalls = torch.sum(hit, 2).double().view(-1)
        recalls.div_(gt_boxes.size(0))

        for n in ns:
            key = f'{thresh:.2f}_recall_at_{n}'
            if n <= recalls.size(0):
                stats[key] = recalls[n]

    return stats


def merge_boxes(boxes, thr):
    assert thr > 0
    ix = []
    D = torchvision.ops.box_iou(boxes, boxes)
    while True:
        good = torch.ge(D, .7)
        good_sum = torch.sum(good, 0).view(-1)
        topnum,topix = torch.max(good_sum, dim=0)
        if topnum.item() == 0:
            break
        mergeix = torch.nonzero(good[topix]).view(-1)

        ix.append(mergeix)
        D.index_fill_(0, mergeix, 0)
        D.index_fill_(1, mergeix, 0)

    return ix

