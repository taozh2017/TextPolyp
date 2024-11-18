import os
import pdb
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import *
import cv2
import torch
import collections

import pdb
import sys
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from lib import dataset as val_dataset
from lib.CEANet import Net
import logging as logger
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from lib.tools import *
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
import json
from typing import Any, Dict, List
from segment_anything.util import get_boxes_from_mask


# text_prompt="A polyp is an anomalous ovalshaped small bump-like structure, a relatively small growth or mass that develops on the inner lining of the colon or other organs. Multiple polyps may exist in one image.Appearance of polyps In most cases, they appear as protruding, visible lumps or nodules. The color may vary and may usually be pink or pale yellow.",
# text_prompt="A polyp is an abnormal, irregularly shaped, lump-like structure that is a growth or mass that forms on the lining of the colon or other organs. The appearance of polyps in most cases appears as prominent, visible bumps or nodules. The color may vary, usually pink or pale yellow, with no deep black gray.",
# text3
# text_prompt='A polyp is an abnormal, irregularly shaped, lumpy structure that is a growth or lump. The appearance of polyps in most cases appears as prominent, visible bumps or nodules. The color is usually pink or pale yellow, not dark black gray.',
# text4
# Polyps are abnormal growths that start in the inner lining of the colon or rectum. Some polyps are flat while others have a stalk. A colorectal polyp is a growth on the lining of the colon or rectum. Adenomatous polyps are a common type.  They are gland-like growths that develop on the mucous membrane that lines the large intestine.  They are also called adenomas and are most often one of the following:Tubular polyp, which protrudes out in the lumen (open space) of the colon,Villous adenoma, which is sometimes flat and spreading, Hyperplastic polyps,Serrated polyps, which are less common.
# text5
# A colorectal polyp is abnormal growths on the lining of the colon or rectum. Some polyps are flat while others have a stalk. Polyps come in a variety of shapes, which is sometimes flat and round,and most of the time irregular. The color of the polyp is not very different from the surrounding normal tissue, and the polyp has no distinct boundary.


def calculate_box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def filter_boxes(boxes, point):
    filtered_boxes = []
    for box in boxes:
        if point[0] >= box[0] and point[1] >= box[1] and point[0] <= box[2] and point[1] <= box[3]:
            filtered_boxes.append(box)
    return filtered_boxes


import torch


def expand_point_to_box(x, y, expansion):
    x1 = max(x - expansion, 0)
    y1 = max(y - expansion, 0)
    x2 = max(x + expansion, 0)
    y2 = max(y + expansion, 0)
    # box_tensor = torch.tensor([[x1, y1, x2, y2]])
    box = np.array([[x1, y1, x2, y2]])
    return box


def process_boxes(boxes, image, name):
    image_area = image[0] * image[1]
    N = boxes.shape[0]
    boxss = np.array([])
    if N == 1:
        box_area = calculate_box_area(boxes)
        if box_area > 0.7 * image_area:
            return 0, None
        else:
            boxss = np.append(boxss, boxes.numpy())
            return 1, boxss
    else:
        min_box_index = torch.argmin(calculate_box_area(boxes))
        min_box_area = calculate_box_area(boxes[min_box_index:min_box_index + 1])
        tensor = torch.min(boxes[:, 2:], boxes[min_box_index, 2:]) - torch.max(boxes[:, :2], boxes[min_box_index, :2])
        row_products = tensor[:, 0] * tensor[:, 1]
        intersection_area = torch.min(row_products)
        if intersection_area > 0.8 * min_box_area:
            result_box = boxes[min_box_index:min_box_index + 1]
            boxss = np.append(boxss, result_box.numpy())
        else:
            result_box = torch.cat([
                torch.min(boxes[:, :2], dim=0).values,
                torch.max(boxes[:, 2:], dim=0).values
            ]).view(1, 4)
            if result_box.numel() != 0:
                for ia in range(len(result_box)):
                    boxss = np.append(boxss, result_box[ia])
        return 1, boxss


def process_image(name, image_path, point_path, model, predictor,
                  text_prompt='A colorectal polyp is an abnormal growth on the lining of the colon or rectum. Some polyps are flat while others have a stalk. Polyps come in various shapes, which are sometimes flat and round, yet usually irregular. The color of the polyp is similar to the surrounding normal tissue, and the polyp lacks a distinct boundary.',box_threshold=0.25, text_threshold=0.25):
    image_source, image_tensor, image = load_image(image_path)
    image_sam = cv2.imread(image_path)

    image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_sam)
    img_point = cv2.imread(os.path.join(point_path, name))
    # target_color = np.array([0, 255, 0])
    # img_point = np.where(np.all(img_point == target_color, axis=-1, keepdims=True), [0, 0, 0], img_point)
    img_point = img_point[:, :, 0]

    img_xy = np.unravel_index(np.argmax(img_point), img_point.shape)
    input_point = np.array([[img_xy[1], img_xy[0]]])
    # print(input_point)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    polyp_indices = [i for i, phrase in enumerate(phrases) if 'polyp' in phrase.lower()]
    polyp_boxes = boxes[polyp_indices]
    polyp_phrases = [phrases[i] for i in polyp_indices]
    log_path = os.path.join("./DINO2sam", "log_dino.log")
    with open(log_path, 'a') as log:
        log.write(f"Image: {os.path.basename(image_path)}\n")
        for i, (box, phrase) in enumerate(zip(polyp_boxes, polyp_phrases)):
            log.write(f"Poly {i + 1} - Box Coordinates: {box}\n")
            log.write(f"Phrase: {phrase}\n")
        log.write("\n")
        image_size = image_tensor.shape[1:]
        polyp_boxes = torch.tensor(filter_boxes(polyp_boxes, (img_xy[1], img_xy[0])))
        if polyp_boxes.shape[0] == 0:
            # detection none box
            print(name, img_xy[1], img_xy[0])
            box = expand_point_to_box(img_xy[1], img_xy[0], 70)
        else:
            annotated_frame, xyxy = annotate(image_source=image_source, boxes=polyp_boxes, logits=logits[polyp_indices],
                                             phrases=polyp_phrases)
            xyxy[xyxy < 0] = 0
            log.write(f"polyp_boxes: {xyxy}\n")
            flag, box = process_boxes(torch.from_numpy(xyxy), image_size, name)
            if flag == 0:
                box = expand_point_to_box(img_xy[1], img_xy[0], 70)
        log.write(f"box: {box}\n")
        # input_point = np.array([img_xy])
        input_label = np.array([1])
        transformed_boxes = box
        print("name:", name)
        print("point:", input_point)
        print("box:", transformed_boxes)
        print("shape", image_sam.shape)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=transformed_boxes,
            multimask_output=False,
        )
        for i in range(masks.shape[0]):
            pred = masks[i]
            # cv2.imwrite('./out/129/all/' + name[:-4] + '_' + str(i) + '.png', pred * 255)
            cv2.imwrite('./out/pseudo/' + name, pred * 255)
            # cv2.imwrite(output_path + '/' + filename, pred.cpu().numpy() * 255)


def main():
    config_path = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "./groundingdino_swint_ogc.pth"

    config = SLConfig.fromfile(config_path)
    model = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    adjusted_state_dict = collections.OrderedDict()
    prefix_to_remove = 'module.'

    for key, value in checkpoint['model'].items():
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]  # Remove the prefix
            adjusted_state_dict[new_key] = value
        else:
            adjusted_state_dict[key] = value

    # Assuming these are the unwanted keys
    unwanted_keys = ["label_enc.weight", "bert.embeddings.position_ids"]

    # Remove unwanted keys from the adjusted state dictionary
    adjusted_state_dict = {k: v for k, v in adjusted_state_dict.items() if k not in unwanted_keys}

    # Load the adjusted state dictionary into your model
    model.load_state_dict(adjusted_state_dict)

    model.eval()

    # Directory containing all images
    image_dir = "./data/TrainDB/image"
    img_point = './data/TrainDB/point'
    print("Loading sam model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.cuda()
    predictor = SamPredictor(sam)

    output_path = os.path.join("./out/pseudo", 'seg')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sam_path = os.path.join("./sam", 'seg')
    if not os.path.exists(sam_path):
        os.makedirs(sam_path)
    with torch.no_grad():
        for filename in os.listdir(image_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(image_dir, filename)
                if filename != "":

                    process_image(filename, image_path, img_point, model, predictor)


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
parser.add_argument(
    "--model-type",
    type=str,
    default="vit_b",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

if __name__ == "__main__":
    args = parser.parse_args()
    args.checkpoint = './checkpoints/sam_vit_b_01ec64.pth'
    main()
