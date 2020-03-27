import os
from pathlib import Path

import cv2
import sys
import pandas as pd
import json
import csv
from detectron2.structures import BoxMode


def get_deepfashion2_dicts(img_dir):
    filenames = [i.split('.')[0] for i in os.listdir(img_dir + '/annos')]
    dataset_dicts = []
    for i, name in enumerate(filenames):
        record = {}
        image_file = os.path.join(img_dir, 'image/', name + '.jpg')
        height, width = cv2.imread(image_file).shape[:2]

        record["file_name"] = image_file
        record["height"] = height
        record["width"] = width

        objs = []
        json_file = os.path.join(img_dir, 'annos/', name + '.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)
        record["image_id"] = imgs_anns['pair_id']

        items = [i for i in imgs_anns.keys() if 'item' in i]
        for item in items:
            anns_item = imgs_anns[item]
            obj = {
                "bbox": anns_item['bounding_box'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": anns_item['segmentation'],
                "category_id": anns_item['category_id'],
                "iscrowd": 0,
                "labels": anns_item['category_name']
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_imaterialist_dicts(csvname):
    pass



if __name__ == "__main__":
    csv.field_size_limit(9999999)

