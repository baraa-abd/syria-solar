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
import math

# ==========================================================================================
# Utility Functions
# ==========================================================================================

def polygon_to_mask(polygon, width, height):
    """
    Converts a COCO-style polygon segmentation to a binary mask.
    The polygon is a flat list of [x1, y1, x2, y2, ...].
    """
    img = Image.new('L', (width, height), 0)
    if polygon:
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def rle_to_mask(rle_counts, height, width):
    """
    Converts a COCO-style Run-Length Encoding (RLE) to a binary mask.

    Args:
        rle_counts (list): A list of integers representing the RLE.
        height (int): The height of the mask.
        width (int): The width of the mask.

    Returns:
        np.ndarray: A 2D numpy array representing the binary mask.
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    start = 0
    val = 1
    for count in rle_counts:
        mask[start:start + count] = val
        start += count
        val = 1 - val # Alternate between 0 and 1
    return mask.reshape((height, width))

def mask_to_rle(mask):
    """
    Converts a binary mask numpy array to a COCO-style Run-Length Encoding (RLE).
    """
    # Flatten the mask in column-major order (Fortran-style)
    pixels = mask.flatten(order='F')
    # Use groupby to find consecutive runs of the same value
    counts = [sum(1 for _ in group) for _, group in groupby(pixels)]
    # The first count is always for the value 0
    if pixels[0] == 1:
        counts.insert(0, 0)
    return counts

def bbox_xywh2xyxy(bbox):
    """
    Converts a bounding box from [x, y, w, h] to [x_min, y_min, x_max, y_max] format.
    """
    x, y, w, h = bbox
    return [x, y, x+w, y+h]

def bbox_xyxy2xywh(bbox):
    """
    Converts a bounding box from [x_min, y_min, x_max, y_max] to [x, y, w, h] format.
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2-x1, y2-y1]

def bbox_visualize_predictions(model: torch.nn.Module,
                          device: torch.device,
                          image_dir: str,
                          n1: int,
                          n2: int,
                          bbox_transform: transforms.Compose = transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                              ]),
                          threshold: float = 0.7):
    """
    Loads images, runs them through the model, and visualizes the predictions.

    Args:
        model (torch.nn.Module): The trained object detection model.
        device (torch.device): The device to run inference on.
        image_dir (str): Directory containing the images to test.
        n1 (int): The starting number for the image files.
        n2 (int): The ending number for the image files.
        threshold (float): The confidence score threshold for displaying predictions.
    """
    print("\n--- Starting Visualization ---")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for i in range(n1, n2 + 1):
            img_path: str = os.path.join(image_dir, f"img_{i}.png")
            if not os.path.exists(img_path):
                print(f"Skipping... Image not found: {img_path}")
                continue

            print(f"Processing {img_path}...")
            image: Image.Image = Image.open(img_path).convert("RGB")

            # Prepare the image for the model
            img_tensor: torch.Tensor = bbox_transform(image).to(device)

            # Get predictions
            prediction: typing.Dict[str, torch.Tensor] = model([img_tensor])

            # prediction is a list of dicts. For a single image, it's prediction[0].
            pred_data: typing.Dict[str, torch.Tensor] = prediction[0]

            # Filter predictions based on the score threshold
            pred_boxes: torch.Tensor = pred_data['boxes'][pred_data['scores'] > threshold]

            # Draw boxes on the original image
            draw = ImageDraw.Draw(image)
            for box in pred_boxes:
                box: np.ndarray = box.cpu().numpy()
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

            # Display the image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.title(f"Predictions for img_{i}.png (Threshold: {threshold})")
            plt.axis('off')
            plt.show()

def overlay_annotations(image, masks=None, bboxes=None, keypoints=None, save_suffix = "", show = True):
    """
    Overlays annotations on an image and displays it.

    Args:
        image (PIL.Image or np.ndarray or torch.Tensor): The original image.
        masks (np.ndarray or list of np.ndarray): A binary segmentation mask or a list of masks of size (H, W).
        bboxes (list, optional): List of bounding boxes to draw in (x,y,w,h) format or single bounding box.
        keypoints (list, optional): A list [x1,y1,x2,y2] or a list of such lists to plot as points.
        save_suffix (string, optional): String suffix for the saved file name.
        show (Boolean, optional): Boolean determining whether each image is shown.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Create a colored overlay from the mask if provided
    # The color is green with ~60% opacity
    if masks is not None:
        if not isinstance(masks, list): masks = [masks]
        for m in masks:
            overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
            overlay[m > 0] = [0, 255, 0, 150]  # RGBA for green
            ax.imshow(overlay)

    # Draw bounding boxes if provided
    if bboxes:
        if isinstance(bboxes[0], (int, float)):
            bboxes = [bboxes]
        for bbox in bboxes:
            if len(bbox)!=4: continue
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    # Plot keypoints if provided
    if keypoints is not None:
        # Ensure keypoints is a list of lists for uniform processing
        if isinstance(keypoints[0], (int, float)):
            keypoints = [keypoints]
        for kp_set in keypoints:
            if len(kp_set) != 4: continue
            x1, y1, x2, y2 = kp_set
            ax.scatter(x1, y1, color='blue', marker='o', s=100, label='Point 1') # Plot (x1, y1) as a blue circle
            ax.scatter(x2, y2, color='yellow', marker='*', s=100, label='Point 2') # Plot (x2, y2) as a yellow star

    title_parts = []
    if mask is not None: title_parts.append("mask overlay")
    if bboxes is not None: title_parts.append(f"{len(bboxes)} bounding boxes")
    if keypoints is not None: title_parts.append(f"{len(keypoints)} keypoint sets")
    
    ax.set_title(f"Image with {', '.join(title_parts) if title_parts else 'No Annotations'}")
    ax.axis('off')
    plt.tight_layout()

    if save_suffix!="":
      plt.savefig(f"img{save_suffix}.png")
      print(f"Image with annotations saved to 'img{save_suffix}.png'")
    if show: plt.show()

def calculate_val_loss_bbox(model, val_loader, device):
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            if not images: continue
            loss_dict = model([img.to(device) for img in images], [{k: v.to(device) for k, v in t.items()} for t in targets])
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            del images, targets, losses, loss_dict
    return val_loss

def calculate_val_loss_corner(model, val_loader, criterion, device):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            if data[0] is None: continue
            images, bboxes, masks, keypoints = data
            outputs = model(images.to(device), bboxes.to(device), masks.to(device))
            val_loss += criterion(outputs, keypoints.to(device)).item()
            del data, outputs
    return val_loss

def calculate_binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    else:
        # Calculate entropy in bits (using log base 2)
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    
def get_top_k_entropy_images(scores_list: typing.List[typing.Dict[str, typing.Union[int, float, str]]], k: int = 10) -> typing.List[typing.Tuple[float, typing.Union[int, str]]]:
    """
    Calculates the classification entropy for a list of scores and returns the top-k with the highest entropy.

    Args:
        scores_list (list): A list of dictionaries, where each dictionary has 'file_name' and 'score' keys.
        k (int): The number of top predictions to return.

    Returns:
        top_k_entropies: A list of tuples (entropy, file_name) for the top k predictions with highest entropy.
              Returns an empty list if no scores are provided or an error occurs.
    """
    entropies: typing.List[typing.Tuple[float, typing.Union[int, str]]] = []

    for score_entry in scores_list:
        score: typing.Optional[typing.Union[int, float]] = score_entry.get('score')
        file_name: typing.Union[int, str] = score_entry.get('file_name', 'N/A')

        if score is not None and isinstance(score, (int, float)):
            entropies.append((calculate_binary_entropy(score), file_name))
        else:
            print(f"Warning: Score entry for image {file_name} has invalid score: {score}. Skipping.")

    entropies.sort(key=lambda item: item[0], reverse=True)
    return entropies[:k]

# ==========================================================================================
# Dataset Classes
# ==========================================================================================

class BoundingboxDataset(Dataset):
    """
    Dataset for the Bounding Box detection task.
    Each item is a full image and a target dictionary containing all bounding boxes for that image.
    """
    def __init__(self, json_file, image_dir, transform=None, image_ids=None, return_ids = False, only_pos = False, verbose = False):
        self.image_dir = image_dir
        self.transform = transform
        self.return_ids = return_ids
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        self.image_info_map = {img['id']: img for img in coco_data['images']}

        # Create a mapping from image_id to annotations
        self.annotations_by_image = {img_id: [] for img_id in self.image_info_map.keys()}
        for ann in coco_data['annotations']:
            if ann.get('iscrowd') == 0:
                self.annotations_by_image[ann['image_id']].append(ann)

        # Filter image_ids to include only those provided and if only_pos is True only those with annotations
        if image_ids is None: image_ids = self.annotations_by_image.keys()
        self.image_ids = [img_id for img_id in self.annotations_by_image if img_id in image_ids and (not only_pos or self.annotations_by_image[img_id])]

        if verbose: print(f"Initialized BoundingboxDataset with {len(self.image_ids)} images:  Subset size:{len(image_ids)}/{len(self.image_info_map.keys())}    Only including positive examples: {only_pos}.")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_data = self.image_info_map[image_id]
        image_path = os.path.join(self.image_dir, image_data['file_name'])
        image = Image.open(image_path).convert("RGB")

        annotations = self.annotations_by_image[image_id]
        boxes, labels = [], []
        for ann in annotations:
            boxes.append(bbox_xywh2xyxy(ann['bbox']))
            labels.append(ann.get('category_id', 1))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes)>0 else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return (image, target, image_id) if self.return_ids else (image, target)


class SinglePanelDataset(Dataset):
    """
    Dataset for tasks involving single solar panel instances (SAM, Corner Prediction).
    Each item corresponds to a single solar panel annotation.
    """
    def __init__(self, json_file, image_dir, image_transform=None, mask_transform=None, keypoint_transform=None, image_ids=None, verbose = False):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.keypoint_transform = keypoint_transform
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        self.image_info = {img['id']: img for img in coco_data['images']}

        all_annotations = [ann for ann in coco_data['annotations'] if ann.get('iscrowd') == 0 and 'bbox' in ann and ann['bbox']]

        if image_ids is not None:
            image_ids_set = set(image_ids)
            self.annotations = [ann for ann in all_annotations if ann['image_id'] in image_ids_set]
        else:
            self.annotations = all_annotations
        self.img2ann = {image_id: [ann for ann in self.annotations if ann['image_id']==image_id] for image_id in self.image_info.keys() if image_ids is None or image_id in image_ids}
        if verbose: print(f"Initialized SinglePanelDataset with {len(self.annotations)} annotations on {len(image_ids) if image_ids is not None else 'all'} images.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.process_annotation(self.annotations[idx])

    def process_annotation(self, ann):
        image_id = ann['image_id']
        image_data = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_data['file_name'])
        image = Image.open(image_path).convert("RGB")

        bbox_xywh = ann['bbox']
        if type(ann.get('segmentation', [[]])[0]) == list:
            if len(ann.get('segmentation', [[]]))>1: print("WARNING: more than one polygon mask in the same annotation, keeping only the first one.")
            mask = polygon_to_mask(ann.get('segmentation', [[]])[0], image_data['width'], image_data['height'])
        else:
            mask = rle_to_mask(ann['segmentation'], image_data['height'], image_data['width'])

        keypoints = None
        if 'keypoints' in ann and len(ann['keypoints']) == 2:
            keypoints_indices = ann['keypoints']
            segmentation_poly = ann.get('segmentation', [[]])[0]
            if segmentation_poly and max(keypoints_indices[0]+1, keypoints_indices[1]+1) < len(segmentation_poly):
                x_bl, y_bl = segmentation_poly[keypoints_indices[0]], segmentation_poly[keypoints_indices[0]+1]
                x_br, y_br = segmentation_poly[keypoints_indices[1]], segmentation_poly[keypoints_indices[1]+1]
                keypoints = torch.tensor([x_bl, y_bl, x_br, y_br], dtype=torch.float32)
        elif 'keypoints' in ann and len(ann['keypoints']) == 4:
            keypoints = torch.tensor(ann['keypoints'], dtype=torch.float32)

        transformed_image = self.image_transform(image) if self.image_transform else image
        transformed_mask = self.mask_transform(Image.fromarray(mask)) if self.mask_transform else torch.from_numpy(mask).float()

        target = {
            "box_xywh": torch.tensor(bbox_xywh, dtype=torch.float32),
            "mask": transformed_mask,
            "image_id": torch.tensor(image_id, dtype=torch.int64)
        }
        if keypoints is not None:
            if self.keypoint_transform:
                keypoints = self.keypoint_transform(keypoints, (width, height))
            target["keypoints"] = keypoints

        return transformed_image, target

def collate_fn_bbox(batch):
    return tuple(zip(*batch))

def collate_fn_corner(batch):
    """Custom collate for corner predictor, handles missing keypoints."""
    batch = list(filter(lambda x: x is not None and "keypoints" in x[1], batch))
    if not batch:
        return None, None, None, None
    images, bboxes, masks, keypoints = [], [], [], []
    for img, target in batch:
        images.append(img)
        bboxes.append(target['box_xywh'])
        masks.append(target['mask'])
        keypoints.append(target['keypoints'])
    return torch.stack(images), torch.stack(bboxes), torch.stack(masks), torch.stack(keypoints)


def dataset_split(dataset, batch_sizes, split_ratio, collate_fn = collate_fn_bbox, shuffle = [True, False], titles = ["Training set", "Validation set"], seed = 42) -> typing.Tuple[typing.Optional[DataLoader], typing.Optional[DataLoader]]:
    """
    Splits dataset and sets up dataloaders.

    Args:
        dataset (str): Path to the COCO format JSON file.
        batch_sizes (list): list of 2 integers indicating batch sizes for each dataloader.
        split_ratio (float): Fraction of the dataset in first part.
        collate_fn (function): collate function to be used.
        shuffle (list of bools): list of 2 booleans indicating whether dataloader is shuffled for each split.
        titles (list of str): two strings indicating title for each split.
        seed (int): seed for random split
    Returns:
        dataloader1, dataloader2
    """
    size1: int = int(len(dataset) * split_ratio)
    size2: int = len(dataset) - size1
    dataset1, dataset2 = random_split(dataset, [size1, size2], generator=torch.Generator().manual_seed(seed))

    print(f"{titles[0]} size: {size1}")
    print(f"{titles[1]} size: {size2}")

    dataloader1 = DataLoader(
        dataset1, batch_size=batch_sizes[0], shuffle=shuffle[0],
        num_workers=2, collate_fn=collate_fn
    )

    dataloader2 = DataLoader(
        dataset2, batch_size=batch_sizes[1], shuffle=shuffle[1],
        num_workers=2, collate_fn=collate_fn
    )

    return dataloader1, dataloader2

# ==========================================================================================
# Models
# ==========================================================================================

def get_bbox_detection_model(num_classes):
    """Loads a pre-trained Faster R-CNN model and adapts it for our number of classes."""
    model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    # num_classes includes the background, so for solar panels it's 2 (background + solar_panel)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def load_bbox_model(device, path = None, num_classes=2):
    """Loads a trained Faster R-CNN model from a state dictionary file."""
    model = get_bbox_detection_model(num_classes)
    if path is not None: model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Bounding box model loaded from {path}")
    return model

def load_sam_model(path_or_checkpoint, device):
    """Loads a fine-tuned SAM model and its processor from a directory."""
    model = SamModel.from_pretrained(path_or_checkpoint)
    processor = SamProcessor.from_pretrained(path_or_checkpoint)
    model.to(device)
    model.eval()
    print(f"SAM model loaded from {path_or_checkpoint}")
    return model, processor

class CornerPredictor(nn.Module):
    """The main model for predicting the corners of a solar panel."""
    def __init__(self, device, feature_extractor_name='resnet50', pretrained=True, backbone=None, strategy = "basic"):
        super(CornerPredictor, self).__init__()
        if backbone:
            print("Using provided backbone for feature extraction.")
            self.feature_extractor = backbone
        else:
            self.feature_extractor = timm.create_model(feature_extractor_name, pretrained=pretrained, num_classes=0, global_pool='').to(device)
        self.strategy = strategy

        output_size = 7
        self.roi_pooler = ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=output_size,
            sampling_ratio=2
        )

        self.img_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * output_size * output_size, 512),
            nn.ReLU()
        )

        self.bbox_processor = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.mask_processor = nn.Sequential(
            # Input: 1 x 224 x 224
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 16 x 112 x 112
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 56 x 56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 28 x 28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 14 x 14
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # Output: 64
        )
        if strategy == "basic":
            self.combiner = nn.Sequential(
                nn.Linear(512 + 32 + 64, 1024),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 4)
            )
        elif strategy == "attention":
            # Cross-Attention Combiner
            query_dim = 32 + 64
            feature_dim = 256
            self.attention = nn.MultiheadAttention(
                embed_dim=query_dim,
                num_heads=8,
                kdim=feature_dim,
                vdim=feature_dim,
                batch_first=True
            )
            self.combiner = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
            )
        elif strategy == "crop":
            self.combiner = nn.Sequential(
                nn.Linear(1024 + 32 + 64, 2048),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 4)
            )

    def forward(self, image, bbox, mask):
        scale = torch.tensor([image.shape[-1], image.shape[-2]] * 2, dtype=torch.float32).to(image.device)
        image_features = self.feature_extractor(image)
        roi_features = self.roi_pooler(image_features, [bbox], [image.shape[-2:]])
        bbox_emb = self.bbox_processor(bbox)
        mask_emb = self.mask_processor(mask)
        if self.strategy == "basic":
            img_emb = self.img_processor(roi_features)           
            return (self.combiner(torch.cat([img_emb, bbox_emb, mask_emb], dim=1))+1.0)*scale
        elif self.strategy == "attention":
            img_seq = roi_features.flatten(2).permute(0, 2, 1)
            query = torch.cat([bbox_emb, mask_emb], dim=1).unsqueeze(1)
            attn_output, _ = self.attention(query=query, key=img_seq, value=img_seq)
            return (self.combiner(attn_output.squeeze(1))+1.0)*scale
        elif self.strategy == "crop":
            cropped_img = torch.zeros_like(image)
            for i in range(image.shape[0]):
                x, y, w, h = bbox[i].int()
                cropped_img[i, :, y:y+h, x:x+w] = image[i, :, y:y+h, x:x+w]
            cropped_features = self.feature_extractor(cropped_img)
            cropped_roi_features = self.roi_pooler(cropped_features, [bbox], [cropped_img.shape[-2:]])
            cropped_emb = self.img_processor(cropped_roi_features)
            img_emb = self.img_processor(roi_features)
            return (self.combiner(torch.cat([img_emb, cropped_emb, bbox_emb, mask_emb], dim=1))+1.0)*scale


def load_corner_model(path, backbone, device, strategy = 'basic'):
    """Loads a trained CornerPredictor model from a state dictionary file."""
    model = CornerPredictor(device, backbone = backbone, strategy = strategy)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Corner predictor model loaded from {path}")
    return model

# ==========================================================================================
# Loss functions
# ==========================================================================================

class CombinedLoss(nn.Module):
    """
    Combines Dice Loss and Binary Cross-Entropy Loss.
    This is a common loss function for segmentation tasks.
    """
    def __init__(self, weight_dice=0.8, weight_bce=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets, smooth=1):
        dice = self.dice_loss(inputs, targets, smooth)
        bce = self.bce_loss(inputs, targets)
        return self.weight_dice * dice + self.weight_bce * bce

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# ==========================================================================================
# Training functions
# ==========================================================================================

def train_corner_model_one_step(corner_model, corner_loader, corner_optimizer, corner_criterion, val_corner_loader, device):
    train_loss_corner = 0
    corner_model.train()
    for data in corner_loader:
        if data[0] is None: continue
        images, bboxes, masks, keypoints = data
        outputs = corner_model(images.to(device), bboxes.to(device), masks.to(device))
        loss = corner_criterion(outputs, keypoints.to(device))
        train_loss_corner += loss.item()
        corner_optimizer.zero_grad(set_to_none = True); loss.backward(); corner_optimizer.step()
        del images, bboxes, masks, keypoints, outputs, loss, data
    val_loss_corner = calculate_val_loss_corner(corner_model, val_corner_loader, corner_criterion, device)
    return train_loss_corner, val_loss_corner

def train_corner_model(corner_model, epochs, corner_loader, corner_optimizer, corner_criterion, val_corner_loader, device, verbose = False):
    val_losses_corner = []
    train_losses_corner = []
    for epoch in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print('-' * 25)
        train_loss_corner, val_loss_corner = train_corner_model_one_step(corner_model, corner_loader, corner_optimizer, corner_criterion, val_corner_loader, device)
        val_losses_corner.append(val_loss_corner/len(val_corner_loader))
        train_losses_corner.append(train_loss_corner/len(corner_loader))
        if verbose: print(f"Corner Head - Epoch {epoch+1}, Train Loss: {train_losses_corner[-1]:.4f}, Val Loss: {val_losses_corner[-1]:.4f}")
    return train_losses_corner, val_losses_corner

def train_bbox_corner_together(bbox_model, corner_model, epochs, bbox_loader, corner_loader, bbox_optimizer, corner_optimizer, corner_criterion, val_bbox_loader, val_corner_loader, device, verbose = False):
    val_losses_bbox = []
    val_losses_corner = []
    train_losses_bbox = []
    train_losses_corner = []
    for epoch in range(epochs):
        if verbose:
          print(f"\nEpoch {epoch+1}/{epochs}")
          print('-' * 25)
        train_loss_bbox = 0
        bbox_model.train()
        for images, targets in bbox_loader:
            if not images: continue
            loss_dict = bbox_model([img.to(device) for img in images], [{k: v.to(device) for k, v in t.items()} for t in targets])
            losses = sum(loss for loss in loss_dict.values())
            train_loss_bbox += losses.item()
            bbox_optimizer.zero_grad(set_to_none = True); losses.backward(); bbox_optimizer.step()
            del images, targets, loss_dict, losses

        val_loss_bbox = calculate_val_loss_bbox(bbox_model, val_bbox_loader, device)
        val_losses_bbox.append(val_loss_bbox/len(val_bbox_loader))
        train_losses_bbox.append(train_loss_bbox/len(bbox_loader))

        train_loss_corner, val_loss_corner = train_corner_model_one_step(corner_model, corner_loader, corner_optimizer, corner_criterion, val_corner_loader, device)
        val_losses_corner.append(val_loss_corner/len(val_corner_loader))
        train_losses_corner.append(train_loss_corner/len(corner_loader))
        if verbose:
          print(f"BBox Head - Epoch {epoch+1}, Train Loss: {train_losses_bbox[-1]:.4f}, Val Loss: {val_losses_bbox[-1]:.4f}")
          print(f"Corner Head - Epoch {epoch+1}, Train Loss: {train_losses_corner[-1]:.4f}, Val Loss: {val_losses_corner[-1]:.4f}")
    return train_losses_bbox, train_losses_corner, val_losses_bbox, val_losses_corner

def fine_tune_sam_model(model, processor, train_loader, val_loader, device, num_epochs=5, learning_rate=1e-5, save_path=None):
    """Fine-tunes the SAM model on individual solar panels."""
    model.to(device)
    model.train()

    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("mask_decoder"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = CombinedLoss()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            if not batch: continue
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_mask = batch["ground_truth_mask"].to(device)

            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False) #this only takes best guess mask
            predicted_masks = outputs.pred_masks.squeeze(1)
            loss = loss_fn(predicted_masks, ground_truth_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                pixel_values = batch["pixel_values"].to(device)
                input_boxes = batch["input_boxes"].to(device)
                ground_truth_mask = batch["ground_truth_mask"].to(device)
                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)
                loss = loss_fn(predicted_masks, ground_truth_mask)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_path:
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"Model saved to '{save_path}'")
            
# ==========================================================================================
# Inference
# ==========================================================================================

def run_bbox_model_single(bbox_model, image, bbox_transform, device, threshold = 0.7):
    bbox_model.eval()
    with torch.no_grad():
        # Prepare image for model input
        img_tensor = bbox_transform(image).to(device)
        # Get predictions
        prediction = bbox_model([img_tensor])[0]
        scores: torch.Tensor = prediction['scores']
        boxes: torch.Tensor = prediction['boxes']
        labels: torch.Tensor = prediction['labels']
        return [(scores[i], boxes[i], labels[i]) for i in range(len(scores)) if scores[i]>=threshold]

def run_bbox_model(model: torch.nn.Module, bbox_transform, image_dir: str, device: torch.device, in_subset = None, banned_subset = None, threshold = 0.7, output_json_path = None, verbose = False) -> typing.Dict[str, typing.Any]:
                            
    """
    Generates a COCO-formatted dictionary with the model's bounding box predictions on images in image directory (within in_subset but not in banned_subset). 

    Args:
        model (torch.nn.Module): The trained object detection model.
        bbox_transform: transform to apply on the image. 
        image_dir (str): Directory containing the images to process. Image filenames must end with .png, .jpg, or .jpeg, and have a number preceding the extension.
        device (torch.device): The device to run inference on.
        in_subset (set): set of filenames of images to process (if None all images in the directory not in banned_subset are processed). 
        banned_subset (set): set of filenames of images to skip (if None all images in the directory in in_subset are processed). 
        threshold (float): The confidence score threshold for saving predictions.
        output_json_path (str): Path to save the generated JSON file.

    Returns:
        coco_output (dict): A dictionary containing the COCO-formatted predictions.
    """
    print(f"\n--- Running bounding box model ---")
    model.to(device)
    model.eval()

    coco_output: typing.Dict[str, typing.Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "solar_panel", "supercategory": "object"}]
    }

    annotation_id_counter = 1
    image_id_counter = 1
    
    image_files = sorted([f for f in os.listdir(image_dir)
                if (f.lower().endswith(('.png', '.jpg', '.jpeg'))
                and (os.path.splitext(f)[0][-1]!=")")   #to deal with issue on google drive where it duplicates files for some reason (getting "image_N (1).png")
                and (in_subset is None or f in in_subset)
                and (banned_subset is None or f not in banned_subset))])

    with torch.no_grad():
        for image_name in image_files:
            if verbose: print(f"Processing {image_name}...")
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            # Add image info to COCO output
            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": image_name,
                "width": width,
                "height": height
            })

            results = run_bbox_model_single(model, image, bbox_transform, device, threshold = threshold)

            for score, box, label in results:
                coco_bbox = bbox_xyxy2xywh(box.cpu().numpy())
                coco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": label.cpu().item(),
                    "bbox": coco_bbox,
                    "area": float(coco_bbox[2] * coco_bbox[3]),
                    "iscrowd": 0,
                    "score": score.cpu().item()
                })
                annotation_id_counter += 1
            image_id_counter += 1

    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)
        print(f"\nInference complete. Results saved to {output_json_path}")
    return coco_output

def run_sam_model_single(sam_model, sam_processor, image_pil, box_xyxy, device):
    sam_model.eval()
    sam_inputs = sam_processor(image_pil, input_boxes=[[box_xyxy]], return_tensors="pt")
    sam_inputs = {k: v.to(device) for k, v in sam_inputs.items()}
    with torch.no_grad():
        sam_outputs = sam_model(
                pixel_values=sam_inputs["pixel_values"],
                input_boxes=sam_inputs["input_boxes"],
                multimask_output=True
            )
    pred_masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu())[0]

    best_mask_indices = torch.argmax(sam_outputs.iou_scores, dim=-1)[0]
    best_masks = torch.zeros_like(pred_masks[:, 0, :, :]) # Initialize with the correct shape
    for j in range(best_mask_indices.shape[0]):
        best_masks[j] = pred_masks[j, best_mask_indices[j], :, :]
    return best_masks[0]

def run_inference_pipeline_single(bbox_model, sam_model, sam_processor, corner_model, transforms, image_pil, image_name, image_id, device, bbox_threshold=0.7, next_annotation_id = 1):
    """
    Runs the full 3-stage inference pipeline on a single image and returns coco annotations and image info. 

    Args:
        bbox_model: Faster R-CNN model.
        sam_model: SAM model.
        sam_processor: The processor for the SAM model.
        corner_model: CornerPredictor model.
        transforms: a list [bbox_transform, corner_img_transform, corner_mask_transform] of the transforms for the bbox model and corner model.
        image_pil (PIL.Image): a 224 by 224 image
        image_name (str): filename for image
        image_id (int): desired image_id  
        device (torch.device): The device to run inference on.
        bbox_threshold (float): Confidence threshold for bounding box detections.
        next_annotation_id (int): starting annotation id
    """
    
    bbox_transform = transforms[0]
    corner_img_transform = transforms[1]
    corner_mask_transform = transforms[2]

    original_width, original_height = image_pil.size
    annotations = []
    image_info = {"id": image_id,
            "file_name": image_name,
            "width": original_width,
            "height": original_height}

    with torch.no_grad():
        # --- Stage 1: Bounding Box Detection ---
        predicted_boxes = run_bbox_model_single(bbox_model, image_pil, bbox_transform, device, threshold = bbox_threshold)
        for score, bbox_xyxy_tensor, label in predicted_boxes:
            box_xyxy = box_xyxy_tensor.cpu().numpy().tolist()
            box_xywh = bbox_xyxy2xywh(box_xyxy)

            # --- Stage 2: Segmentation ---
            best_mask = run_sam_model_single(sam_model, sam_processor, image_pil, box_xyxy, device).cpu().numpy().astype(np.uint8)
            rle_mask = mask_to_rle(best_mask)

            # --- Stage 3: Corner Prediction ---
            img_tensor_corner = corner_img_transform(image_pil).unsqueeze(0).to(device)
            mask_tensor_corner = corner_mask_transform(Image.fromarray(best_mask)).unsqueeze(0).to(device)
            bbox_tensor_corner = torch.tensor(box_xywh, dtype=torch.float32).unsqueeze(0).to(device)

            pred_keypoints = corner_model(img_tensor_corner, bbox_tensor_corner, mask_tensor_corner)[0]
            pred_keypoints = [pred_keypoints[i].item() for i in range(4)]

            # --- Assemble Annotation ---
            area = int(np.sum(best_mask))
            annotation = {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [round(c, 2) for c in box_xywh],
                "segmentation": rle_mask,
                "area": area,
                "keypoints": [round(p, 2) for p in pred_keypoints]
            }
            annotations.append(annotation)
            next_annotation_id += 1
    return (image_info, annotations, next_annotation_id)

def run_inference_pipeline(bbox_model, sam_model, sam_processor, corner_model, transforms, image_dir, device, output_json_path = None, bbox_threshold=0.7, verbose = False):
    """
    Runs the full 3-stage inference pipeline on a folder of images and returns (and saves if output path assigned) the results.

    Args:
        bbox_model: Faster R-CNN model.
        sam_model: SAM model.
        sam_processor: The processor for the SAM model.
        corner_model: CornerPredictor model.
        transforms: a list [bbox_transform, corner_img_transform, corner_mask_transform] of the transforms for the bbox model and corner model.
        image_dir (str): Path to the folder containing images for inference, assuming all images are 224 by 224.
        device (torch.device): The device to run inference on.
        output_json_path (str): Path to save the output COCO-formatted JSON file.
        bbox_threshold (float): Confidence threshold for bounding box detections.
    """
    print("\n" + "="*50 + "\n--- Starting Inference Pipeline ---\n" + "="*50)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "solar_panel", "supercategory": "object"}]
    }

    next_annotation_id = 1
    image_id_counter = 1

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    with torch.no_grad():
        for image_name in image_files:
            if verbose: print(f"Processing {image_name}...")
            image_path = os.path.join(image_dir, image_name)
            image_pil = Image.open(image_path).convert("RGB")
            image_info, annotations, next_annotation_id = run_inference_pipeline_single(bbox_model, sam_model, sam_processor, corner_model, transforms, 
                                                                                        image_pil, image_name, image_id_counter,
                                                                        device, bbox_threshold=bbox_threshold, next_annotation_id = next_annotation_id)

            # Add image info and annotations to COCO output
            coco_output["images"].append(image_info)
            coco_output["annotations"].extend(annotations)
            image_id_counter += 1

    if output_json_path:
        # Save the final JSON file
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)
        print(f"\nInference complete. Results saved to {output_json_path}")
    return coco_output

# ==========================================================================================
# Plotting Functions
# ==========================================================================================

def plot_losses(losses, save_path_prefix=None):
    """
    Plots training and validation losses for multitask bounding 
     box and corner model training routine a in a 2x5 grid for each phase.

    Args:
        losses (dict): A dictionary containing loss data. Expected keys are:
            'bbox_train', 'bbox_val', 'corner_train', 'corner_val',
            'combined_train_phase2', 'phase_boundaries'.
        save_path_prefix (str, optional): Prefix to save the plot image.
    """
    # Extract loss data
    bbox_train = losses.get('bbox_train', [])
    bbox_val = losses.get('bbox_val', [])
    corner_train = losses.get('corner_train', [])
    corner_val = losses.get('corner_val', [])
    combined_p2 = losses.get('combined_train_phase2', [])
    boundaries = losses.get('phase_boundaries', [])

    if not any([bbox_train, corner_train]) or not boundaries:
        print("Not enough loss data or phase boundaries to plot.")
        return

    total_epochs = len(bbox_train)
    epochs = np.arange(total_epochs)

    # Define phase boundaries
    p_ends = [0] + boundaries
    phase_ranges = [range(p_ends[i], p_ends[i+1]) for i in range(len(p_ends)-1)]
    num_phases = len(phase_ranges)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=False, sharey=False)
    fig.suptitle('Multitask Training Routine', fontsize=20)

    # Define colors for clarity
    colors = {
        'bbox_train': 'darkorange', 'bbox_val': 'sandybrown',
        'corner_train': 'darkcyan', 'corner_val': 'paleturquoise',
        'combined': 'crimson'
    }

    # Loop through each phase to create the plots
    for phase_idx in range(5):
        ax_bbox = axes[0, phase_idx]
        ax_corner = axes[1, phase_idx]

        if phase_idx >= num_phases: # Handle cases with fewer than 5 phases
            ax_bbox.axis('off')
            ax_corner.axis('off')
            continue

        phase_epoch_range = phase_ranges[phase_idx]
        
        # --- Top Row: BBox Model ---
        if phase_idx == 0:
            ax_bbox.text(0.5, 0.5, "No BBox Training", ha='center', va='center', fontsize=12)
            ax_bbox.axis('off')
        else:
            ax_bbox.set_title(f"Phase {phase_idx}: BBox Loss", fontsize=12)
            ax_bbox.plot(phase_epoch_range, bbox_train[phase_epoch_range.start:phase_epoch_range.stop],
                         color=colors['bbox_train'], label='BBox Train Loss')
            if phase_idx != 2: # No validation in Phase 2
                ax_bbox.plot(phase_epoch_range, bbox_val[phase_epoch_range.start:phase_epoch_range.stop],
                             color=colors['bbox_val'], linestyle='--', label='BBox Val Loss')
            ax_bbox.set_ylabel("Loss", color=colors['bbox_train'])
            ax_bbox.tick_params(axis='y', labelcolor=colors['bbox_train'])
            ax_bbox.legend(loc='upper left')
            ax_bbox.grid(True, linestyle=':')

        # --- Bottom Row: Corner Model ---
        ax_corner.set_title(f"Phase {phase_idx}: Corner Loss", fontsize=12)
        ax_corner.plot(phase_epoch_range, corner_train[phase_epoch_range.start:phase_epoch_range.stop],
                       color=colors['corner_train'], label='Corner Train Loss')
        if phase_idx != 2: # No validation in Phase 2
            ax_corner.plot(phase_epoch_range, corner_val[phase_epoch_range.start:phase_epoch_range.stop],
                           color=colors['corner_val'], linestyle='--', label='Corner Val Loss')
        ax_corner.set_ylabel("Loss", color=colors['corner_train'])
        ax_corner.tick_params(axis='y', labelcolor=colors['corner_train'])
        ax_corner.legend(loc='upper left')
        ax_corner.grid(True, linestyle=':')

        # --- Special Handling for Phase 2 Combined Loss ---
        if phase_idx == 2 and combined_p2:
            # Bbox plot with combined loss
            ax_bbox_twin = ax_bbox.twinx()
            ax_bbox_twin.plot(phase_epoch_range, combined_p2, color=colors['combined'], linestyle=':', label='Combined Loss')
            ax_bbox_twin.set_ylabel("Combined Loss", color=colors['combined'])
            ax_bbox_twin.tick_params(axis='y', labelcolor=colors['combined'])
            ax_bbox_twin.legend(loc='upper right')

            # Corner plot with combined loss
            ax_corner_twin = ax_corner.twinx()
            ax_corner_twin.plot(phase_epoch_range, combined_p2, color=colors['combined'], linestyle=':', label='Combined Loss')
            ax_corner_twin.set_ylabel("Combined Loss", color=colors['combined'])
            ax_corner_twin.tick_params(axis='y', labelcolor=colors['combined'])
            ax_corner_twin.legend(loc='upper right')

        # Set common X-axis label
        ax_corner.set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title

    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_loss_plot.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path_prefix}_loss_plot.png")

    plt.show()


def plot_and_save_top_entropies(top_entropies, save_path_prefix = None):
    """
    Plots the top entropies and optionally saves the plot and data to CSV.

    Args:
        top_entropies (list): A list of tuples ('entropy','image_id').
        save_path_prefix (str, optional): Prefix for saving the plot and CSV files.
    """
    # Sort by entropy in decreasing order
    sorted_entropies = sorted(top_entropies, key=lambda x: x[0], reverse=True)

    image_ids = [item[1].split('.')[0].split('_')[1] for item in sorted_entropies]
    entropies = [item[0] for item in sorted_entropies]
    if len(entropies)==0: raise ValueError("Given empty list of entropies")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar([f"img_{id}.png" for id in image_ids], entropies, color='skyblue')
    plt.xlabel("Image ID")
    plt.ylabel("Entropy")
    plt.ylim(entropies[-1]-0.05, entropies[0]+0.05)
    plt.title("Top Image Entropies")
    plt.xticks(rotation=90)
    plt.tight_layout()

    if save_path_prefix:
        # Save plot
        plot_save_path = f"{save_path_prefix}_entropies.png"
        plt.savefig(plot_save_path)
        print(f"Plot saved to: {plot_save_path}")

        # Save data to CSV
        df_entropies = pd.DataFrame(sorted_entropies, columns = ['entropy', 'file_name'])
        csv_save_path = f"{save_path_prefix}_entropies.csv"
        df_entropies.to_csv(csv_save_path, index=False)
        print(f"Entropies data saved to: {csv_save_path}")

    plt.show()