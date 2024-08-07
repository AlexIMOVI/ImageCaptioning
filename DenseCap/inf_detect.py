import torch
import detectron2

import numpy as np
import os, json, cv2, random
from PIL import Image
import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


img = Image.open('../datasets/vg/VG_100K/53.jpg')
img = np.array(img)
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/rpn_R_50_FPN_1x.yaml")
cfg.INPUT_FORMAT = 'RGB'
predictor = DefaultPredictor(cfg)
outputs = predictor(img)
boxes = outputs['proposals']._fields['proposal_boxes'].tensor
