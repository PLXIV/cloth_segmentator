import cv2
import random
import os
import torch
import time
import collections
import re
import json
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, VisImage
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from sklearn.cluster import KMeans
from dict_generator import get_deepfashion2_dicts

setup_logger()


def visualize(im):
    cv2.imshow('ImageWindow', im)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


def declare_datasets(sets, dataset_paths):
    for d in sets:
        DatasetCatalog.register(d, lambda d=d: get_deepfashion2_dicts(dataset_paths + d))
        MetadataCatalog.get(d)  # .set(
        # thing_classes=["short sleeve top", "long sleeve top", "short sleeve outwear",
        #                "long sleeve outwear", "vest", "sling", "shorts", "trousers",
        #                "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
        #                "sling dress"])


def generate_cfg_configuration(train_data, validation_data):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_data,)
    cfg.DATASETS.TEST = (validation_data,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 100000
    cfg.MODEL.ROI_HEADS.BATCH_SIwZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
    cfg.OUTPUT_DIR = '../models/'
    return cfg


def train_model(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer


def load_best_weights(cfg):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    return cfg


def build_predictor(cfg, threshold_score):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_score
    return DefaultPredictor(cfg), cfg


def evaluator(cfg, trainer):
    eval = COCOEvaluator("validation", cfg, False, output_dir="../results/")
    cfg.DATASETS.TRAIN = ()
    val_loader = build_detection_test_loader(cfg, "validation")
    # trainer = DefaultTrainer(cfg)
    inference_on_dataset(trainer.model, val_loader, eval)


def test_single_prediction(im_dir):
    im = cv2.imread(im_dir)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualize(v.get_image()[:, :, ::-1])


def bb_intersection_over_union(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def test_predictor(predictor, dataset, metadata, number_of_samples):
    dataset_dicts = get_deepfashion2_dicts(dataset)
    for d in random.sample(dataset_dicts, number_of_samples):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        print(outputs['instances'].get('pred_classes'))
        v = Visualizer(im[:, :, :],
                       metadata=metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        visualize(v.get_image()[:, :, ::-1])


def get_variables(outputs):
    bboxes = outputs.get('pred_boxes').tensor.tolist()
    predictions = outputs.get('pred_classes').tolist()
    scores = outputs.get('scores').tolist()

    remove_bboxes = []
    if len(bboxes) > 1:
        for i in range(len(bboxes) - 1):
            iou = bb_intersection_over_union(bboxes[i], bboxes[i + 1])
            if iou > 0.8:
                remove_bboxes.append(i + 1)

        for index in sorted(remove_bboxes, reverse=True):
            del bboxes[index]
            del predictions[index]
            del scores[index]

    return bboxes, predictions, scores

'''
def mask_ratio(mask, bbox):
    ratio_im = mask
    ratio_im = ratio_im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    unique, ocurrences = np.unique(ratio_im, return_counts=True)
    unique_dic = dict(zip(unique, ocurrences))
    ratio = unique_dic[1] / unique_dic[0]
    name = ''
    if ratio < 1.0:
        name = 'negative_'
    return name
'''

def store_segmented_image(im, mask, bbox, filename, pred, vis=False):
    """
    This function stores an image with only the cloth that is been segmented

    :param im: hole image where the cloth appears
    :param mask: segmentation mask which allow us to distinguish between foreground and background
    :param bbox: bbox of the cloth that is currently being analyzed
    :param filename: filename of the current image, without the jpg part
    :param pred: name of the class predicted
    :type vis: to show how res would look like
    """
    idx = (mask == 0)
    res = np.copy(im)
    res[idx] = (173., 255., 47.)
    res = res[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    store_image_path = '../segmented_images/images/' + filename + '_' + str(pred) + '.jpg'
    cv2.imwrite(store_image_path, res)
    if vis:
        visualize(res)



def SL_json_extractor(predictor, dataset, folder_name):
    images = os.listdir(dataset)
    final_json = {}
    tmp_jsons = []


    if not os.path.exists('../segmented_images/images/' + folder_name):
        os.makedirs('../segmented_images/images/' + folder_name)

    if folder_name:
        for filename in images:
            im = cv2.imread(dataset + '/' + filename)
            outputs = predictor(im)
            if outputs['instances'].get('pred_boxes'):

                bboxes, predictions, scores = get_variables(outputs['instances'])

                if len(bboxes) > 0:
                    bbox = [int(x) for x in bboxes[0]]
                    mask = outputs['instances'].get('pred_masks')[0].cpu().data.numpy() * 1
                    filename = filename.split('.')[0]

                    store_segmented_image(im, mask, bbox, folder_name + '/' + filename, predictions[0], vis=False)

                    idx = (mask == 1)
                    cloth_segmented = np.copy(im)
                    cloth_segmented = cloth_segmented[idx]
                    kmeans = KMeans(n_clusters=3, random_state=0).fit(cloth_segmented)
                    data = kmeans.predict(cloth_segmented)
                    counter = collections.Counter(data)
                    centroids = kmeans.cluster_centers_

                    colors = obtain_colors(centroids, counter)
                    recommended = {}
                    if '_' not in filename:
                        final_json['idProduct'] = filename
                        final_json['colors'] = colors
                        final_json['type_product'] = predictions[0]
                        final_json['score'] = scores[0]
                        final_json['image_path'] = 'segmented_images/images/' + folder_name + '/' + filename + '.jpg'
                    else:
                        recommended['idProduct'] = filename
                        recommended['colors'] = colors
                        recommended['type_product'] = predictions[0]
                        recommended['score'] = scores[0]
                        recommended['image_path'] = 'segmented_images/images/' + folder_name + '/' + filename + '.jpg'
                    tmp_jsons.append(recommended)

    final_json['related_items'] = tmp_jsons
    json_path = '../segmented_images/jsons/' + folder_name + '.json'
    with open(json_path, 'w') as fp:
        json.dump(final_json, fp)



def json_extractor(predictor, dataset, number_of_samples):

    dataset_dicts = get_deepfashion2_dicts(dataset)
    for d in random.sample(dataset_dicts, number_of_samples):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        if outputs['instances'].get('pred_boxes'):

            bboxes, predictions, scores = get_variables(outputs['instances'])

            for i in range(len(bboxes)):
                bbox = [int(x) for x in bboxes[i]]
                mask = outputs['instances'].get('pred_masks')[i].cpu().data.numpy() * 1
                filename = d["file_name"].split('/image/')[1]
                filename = re.sub('.jpg', '', filename)

                #name = mask_ratio(mask, bbox)
                store_segmented_image(im, mask, bbox, filename, predictions[i], vis=False)

                idx = (mask == 1)
                cloth_segmented = np.copy(im)
                cloth_segmented = cloth_segmented[idx]
                kmeans = KMeans(n_clusters=3, random_state=0).fit(cloth_segmented)
                data = kmeans.predict(cloth_segmented)
                counter = collections.Counter(data)
                centroids = kmeans.cluster_centers_

                colors = obtain_colors(centroids, counter)
                create_json(colors, predictions, scores, filename, i)


def obtain_colors(centroids, counter):
    most_color = counter.most_common(6)
    colors = []
    total_pixels = sum(counter.values())
    #compare_centroids(centroids)
    for sample in most_color:
        center = centroids[sample[0]]
        hex_value = '#{:02x}{:02x}{:02x}'.format(int(center[0]), int(center[1]), int(center[2]))
        percentage = sample[1] / total_pixels
        colors.append([hex_value, percentage])
    return colors

#TODO: in case is necesary, take the centroids and find which ones are similar to each other
def compare_centroids(centroids):
    pass

def create_json(colors, predictions, scores, filename, i):
    new_json = {}
    json_path = '../segmented_images/jsons/' + filename + '_' + str(predictions[i]) + '.json'
    new_json['idProduct'] = filename
    new_json['colors'] = colors
    new_json['type_product'] = predictions[i]
    new_json['score'] = scores[i]
    new_json['image_path'] = 'segmented_images/images/' + filename + '_' + str(predictions[i]) + '.jpg'
    with open(json_path, 'w') as fp:
        json.dump(new_json, fp)


def extract_accuracy(predictor, dataset):
    dataset_dicts = get_deepfashion2_dicts(dataset)
    ypred = []
    yval = []
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        ypred.append(outputs['instances'].get('pred_classes').tolist())
        tmp_yval = []
        for i in d['annotations']:
            tmp_yval.append(i['category_id'])
        yval.append(tmp_yval)

    total_correct = 0
    saying_more_than_required = []
    miss_something = []
    missing_prediction = 0
    wrongly_predicted = 0
    almost_perfectly = 0
    nothingincommon = 0
    partially_classified = 0
    for i in range(len(yval)):
        if set(yval[i][:len(ypred[i])]) == set(ypred[i]):
            total_correct += 1
        else:
            if set(ypred[i]) <= set(yval[i]):
                almost_perfectly += 1
            else:
                union = set(yval[i]) & set(ypred[i])
                if not union:
                    nothingincommon += 1
                    if not yval[i]:
                        missing_prediction += 1
                    else:
                        wrongly_predicted += 1
                else:
                    partially_classified += 1
                    saying_more_than_required.append(abs(len(union) - len(yval[i])))
                    miss_something.append(abs(len(union) - len(ypred[i])))

    print('Perfectly classified: ', str(total_correct / len(yval)))
    print('Almost well classified: ', str(almost_perfectly / len(yval)))
    print('Partially classified: ', str(partially_classified / len(yval)))
    print('More predictions than required (ratio): ',
          str(sum(saying_more_than_required) / len(saying_more_than_required)))
    print('Missing predictions (ratio): ', str(sum(miss_something) / len(miss_something)))
    print('Errors : ', str(nothingincommon / len(yval)))
    print('Nothing predicted: ', str(missing_prediction / len(yval)))
    print('Wrongly classified: ', str(wrongly_predicted / len(yval)))


def test_dataset(img_dir):
    metadata = MetadataCatalog.get(img_dir)
    dataset_dicts = get_deepfashion2_dicts('../data/' + img_dir)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        visualize(vis.get_image()[:, :, ::-1])


def main():
    torch.cuda.empty_cache()

    test_pred = False
    test_dat_load = False
    train = False
    pred = False
    evaluate_bbox = False
    evaluate_predictions = False
    get_jsons = False
    SL_get_jsons = True

    # DeepFashion2 dataset declaration
    DF2sets = ['train', 'validation', 'validation5']
    declare_datasets(DF2sets, '../data/')

    # Shooplook dataset declaration
    SLfolder = '../data/refactored_imgs_top/'
    SLsets = os.listdir(SLfolder)
    declare_datasets(SLsets, SLfolder + '/')


    cfg = generate_cfg_configuration('train', 'validation')

    if test_pred:
        test_single_prediction("../data/input.jpg")

    if test_dat_load:
        test_dataset('validation')

    if train:
        trainer = train_model(cfg)

    if pred:
        load_best_weights(cfg)
        predictor, _ = build_predictor(cfg, 0.4)
        metadata = MetadataCatalog.get('validation5')
        test_predictor(predictor, '../data/validation5', metadata, 5)

    if evaluate_bbox:
        load_best_weights(cfg)
        _, cfg = build_predictor(cfg, 0.4)
        evaluator(cfg, trainer)

    if evaluate_predictions:
        load_best_weights(cfg)
        predictor, _ = build_predictor(cfg, 0.5)
        extract_accuracy(predictor, '../data/validation')

    if get_jsons:
        load_best_weights(cfg)
        predictor, _ = build_predictor(cfg, 0.6)
        json_extractor(predictor, '../data/validation5', 100)

    if SL_get_jsons:
        load_best_weights(cfg)
        predictor, _ = build_predictor(cfg, 0.6)
        for i in SLsets:
            SL_json_extractor(predictor, SLfolder + '/' + i, i)


if __name__ == "__main__":
    main()
