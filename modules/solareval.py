import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops
from transformers import SamModel, SamProcessor
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import json
import os
import timm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from itertools import groupby
import typing
from typing import List
import math
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torchmetrics
import solarutils

# ==========================================================================================
# Evaluation Functions
# ==========================================================================================

def calculate_metrics(bbox_model, sam_model, sam_processor, bbox_transform, image_folder, annotations_json_path, device=None, bbox_threshold = 0.7, sam_threshold = 0.5):
    """
    Evaluates a combined object detection and segmentation model on a dataset with COCO-formatted annotations using the torchmetrics library.

    Args:
        bbox_model: bounding box model, producing a list of bounding boxes (with scores and labels)
        sam_model: segmentation model prompted with a bounding box and producing a mask
        bbox_transform: transform for bounding box model
        image_folder (str): Path to the directory containing the test images.
        annotations_json_path (str): Path to the JSON annotation file.
        device: The device to run the model on.
        bbox_threshold: threshold for bounding box model.
        sam_threshold: threshold for segmentation model.

    Returns:
        dict: A dictionary containing the average IoU, Accuracy, and F1-score
              over the entire dataset, calculated by torchmetrics.
    """
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(annotations_json_path, 'r') as f:
        json_data = json.load(f)
    json_filenames = {image['id']:image['file_name'] for image in json_data['images']}
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f in json_filenames.values()])
    print(f"Loaded {len(image_files)} images from {image_folder} with annotations in {annotations_json_path}.")

    #We collect the annotations into a dictionary image_filename:[annotations]
    image_filename_anns = {}
    for ann in json_data['annotations']:
        image_name = json_filenames[ann['image_id']]
        if ann.get('iscrowd') == 0:
            if image_name not in image_filename_anns: image_filename_anns[image_name] = []
            image_filename_anns[image_name].append(ann)

    # --- Initialize TorchMetrics ---
    # We specify 'binary' task for our semantic mask comparison.
    jaccard = torchmetrics.JaccardIndex(task="binary").to(device)
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)

    # --- Evaluation Loop ---
    print(f"Evaluating model...")
    for image_filename in tqdm(image_files):
        # --- Load Image ---
        img_path = os.path.join(image_folder, image_filename)
        image = Image.open(img_path).convert("RGB")
        
        # Combine all ground truth instance masks into one semantic mask
        combined_true_mask = torch.zeros((224, 224), dtype=torch.bool, device=device)
        if image_filename in image_filename_anns.keys():
            for ann in image_filename_anns[image_filename]:
                if type(ann.get('segmentation', [[]])[0]) == list:
                    if len(ann.get('segmentation', [[]]))>1: print("WARNING: more than one polygon mask in the same annotation, keeping only the first one.")
                    mask = solarutils.polygon_to_mask(ann.get('segmentation', [[]])[0], 224, 224)
                else:
                    mask = solarutils.rle_to_mask(ann['segmentation'], 224, 224)

                combined_true_mask = torch.logical_or(combined_true_mask, torch.from_numpy(mask).to(device).bool())
        
        # --- Get Model Predictions ---
        predicted_boxes = solarutils.run_bbox_model_single(bbox_model, image, bbox_transform, device, threshold = bbox_threshold)
        combined_pred_mask = torch.zeros((224, 224), dtype=torch.bool, device=device)
        for score, bbox_xyxy_tensor, label in predicted_boxes:
            box_xyxy = bbox_xyxy_tensor.cpu().numpy().tolist()
            best_mask = solarutils.run_sam_model_single(sam_model, sam_processor, image, box_xyxy, device)
            best_mask = (best_mask>sam_threshold)
            combined_pred_mask = torch.logical_or(combined_pred_mask, best_mask.to(device).bool())
        
        # --- Update Metrics for the Current Image ---
        jaccard.update(combined_pred_mask, combined_true_mask.int())
        accuracy_metric.update(combined_pred_mask, combined_true_mask.int())
        f1_metric.update(combined_pred_mask, combined_true_mask.int())


    # --- Compute Final Metrics Over All Images ---
    final_metrics = {
        'avg_iou': jaccard.compute().item(),
        'avg_accuracy': accuracy_metric.compute().item(),
        'avg_f1_score': f1_metric.compute().item()
    }

    return final_metrics


def calculate_mAP(model: torch.nn.Module,
                  dataloader: DataLoader,
                  device: torch.device) -> float:
    """
    Calculates the mean average precision of an object detection model on the given dataloader

    Args:
        model (torch.nn.Module): The trained object detection model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device to run inference on.

    Returns:
        float: mAP score
    """
    print("\n--- Calculating mAP ---")
    model.to(device)
    model.eval()

    metric = MeanAveragePrecision(box_format='xyxy')

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            metric.update(predictions, targets)

    result = metric.compute()
    mAP = result['map_50']
    print(f"Mean Average Precision: {mAP:.4f}")
    return mAP