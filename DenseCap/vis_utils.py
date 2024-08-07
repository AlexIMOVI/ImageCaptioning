import torch
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


class VisUtils:
    WAD_COLORS = [
        [173, 35, 25],   # Red
        [42, 75, 215],   # Blue
        [87, 87, 87],    # Dark Gray
        [29, 105, 20],   # Green
        [129, 74, 25],   # Brown
        [129, 197, 122], # Light green
        [157, 175, 255], # Light blue
        [41, 208, 208],  # Cyan
        [255, 146, 51],  # Orange
        [255, 238, 51],  # Yellow
        [233, 222, 187], # Tan
        [255, 205, 243], # Pink
        [0, 0, 0]        # Black
    ]

    @staticmethod
    def clamp(x, low, high):
        return max(min(x, high), low)

    @staticmethod
    def densecap_draw(img, boxes, captions, options=None):
        img = img.clone()
        H, W = img.shape[1], img.shape[2]
        N = boxes.shape[0]

        options = options or {}
        text_size = options.get('text_size', 1)
        box_width = options.get('box_width', 2)

        text_img = torch.zeros_like(img)

        for i in range(N):
            rgb = np.array(VisUtils.WAD_COLORS[i % len(VisUtils.WAD_COLORS)])
            rgb_255 = (255 * rgb).astype(int)
            VisUtils.draw_box(img, boxes[i], rgb, box_width)
            text_opt = {
                'fill': tuple(rgb_255),
                'font': None
            }
            x = boxes[i, 0] + box_width + 1
            y = boxes[i, 1] + box_width + 1
            try:
                draw = ImageDraw.Draw(text_img)
                draw.text((x, y), captions[i], **text_opt)
            except ValueError:
                print('drawText out of bounds: ', x, y, W, H)
        text_img /= 255
        img[text_img != 0] = 0
        img += text_img

        return img

    @staticmethod
    def draw_box(img, box, color, lw=1):
        x, y, w, h = box.tolist()
        H, W = img.shape[1], img.shape[2]

        top_x1 = VisUtils.clamp(x - lw, 1, W)
        top_x2 = VisUtils.clamp(x + w + lw, 1, W)
        top_y1 = VisUtils.clamp(y - lw, 1, H)
        top_y2 = VisUtils.clamp(y + lw, 1, H)

        bottom_y1 = VisUtils.clamp(y + h - lw, 1, H)
        bottom_y2 = VisUtils.clamp(y + h + lw, 1, H)

        left_x1 = VisUtils.clamp(x - lw, 1, W)
        left_x2 = VisUtils.clamp(x + lw, 1, W)
        left_y1 = VisUtils.clamp(y - lw, 1, H)
        left_y2 = VisUtils.clamp(y + h + lw, 1, H)

        right_x1 = VisUtils.clamp(x + w - lw, 1, W)
        right_x2 = VisUtils.clamp(x + w + lw, 1, W)

        for c in range(3):
            cc = color[c] / 255
            img[c, top_y1:top_y2, top_x1:top_x2] = cc
            img[c, bottom_y1:bottom_y2, top_x1:top_x2] = cc
            img[c, left_y1:left_y2, left_x1:left_x2] = cc
            img[c, left_y1:left_y2, right_x1:right_x2] = cc

        return img


vis_utils = VisUtils()
