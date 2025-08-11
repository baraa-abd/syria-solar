import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import sys
import numpy as np
import time
import traceback
from pathlib import Path

# --- Instructions for User ---
# 1. Make sure 'solarutils.py' is in the same directory as this script.
# 2. Create a 'models' directory in the same folder as this script.
# 3. Inside 'models', create two subdirectories: 'bboxes' and 'corners'.
# 4. Place your .pth model files into the respective folders.
# 5. Install required libraries:
#    pip3 install --quiet pandas matplotlib Flask Flask-Cors torch torchvision transformers timm numpy
# 6. Run this script from your terminal:
#    python server.py
# 7. The server will start on http://127.0.0.1:5001.
# ---

# Add the current directory to the path to find solarutils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import solarutils


app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Global Model Cache and Definitions ---
MODEL_CACHE = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_BASE_DIR = Path(__file__).parent / 'models'
print(f"Using device: {DEVICE}")
print(f"Models base directory: {MODELS_BASE_DIR}")


# Define the standard transforms from the notebook
BBOX_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CORNER_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CORNER_MASK_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
TRANSFORMS = [BBOX_TRANSFORM, CORNER_IMG_TRANSFORM, CORNER_MASK_TRANSFORM]


def discover_models():
    """Scans the models directory and returns a list of available model files."""
    models = {'bboxes': [], 'corners': []}
    bbox_dir = MODELS_BASE_DIR / 'bboxes'
    corner_dir = MODELS_BASE_DIR / 'corners'

    if bbox_dir.exists():
        models['bboxes'] = [f.name for f in bbox_dir.glob('*.pth')]
    else:
        print(f"Warning: Directory not found: {bbox_dir}")
        os.makedirs(bbox_dir)

    if corner_dir.exists():
        models['corners'] = [f.name for f in corner_dir.glob('*.pth')]
    else:
        print(f"Warning: Directory not found: {corner_dir}")
        os.makedirs(corner_dir)
        
    return models

def get_models_from_cache(bbox_model_name, corner_model_name, sam_model_name):
    """Loads models from file if not in cache, then returns them."""
    cache_key = f"{bbox_model_name}_{corner_model_name}_{sam_model_name}"
    if cache_key in MODEL_CACHE:
        print(f"Loading models from cache for key: {cache_key}")
        return MODEL_CACHE[cache_key]

    print(f"Cache miss for key: {cache_key}. Loading models from disk...")
    
    bbox_model_path = MODELS_BASE_DIR / 'bboxes' / bbox_model_name
    corner_model_path = MODELS_BASE_DIR / 'corners' / corner_model_name

    if not bbox_model_path.exists() or not corner_model_path.exists():
        raise FileNotFoundError("One or both model files not found on the server.")

    # Load BBox Model from path
    bbox_model = solarutils.load_bbox_model(device=DEVICE, path=str(bbox_model_path))
    print(f"Bounding box model loaded from {bbox_model_path}")

    # Load SAM Model
    sam_model, sam_processor = solarutils.load_sam_model(sam_model_name, DEVICE)
    print("SAM model loaded.")
    
    # Load Corner Model from path
    corner_model = solarutils.load_corner_model(path=str(corner_model_path), backbone=bbox_model.backbone, device=DEVICE, strategy = 'crop')
    print(f"Corner model loaded from {corner_model_path}")
    
    print("All models loaded successfully.")
    
    loaded_models = (bbox_model, sam_model, sam_processor, corner_model)
    MODEL_CACHE[cache_key] = loaded_models
    return loaded_models

@app.route('/models', methods=['GET'])
def get_model_list():
    """Endpoint to get the list of available models."""
    try:
        models = discover_models()
        return jsonify(models)
    except Exception as e:
        print(f"Error discovering models: {e}")
        return f"Error discovering models: {e}", 500

@app.route('/run_inference', methods=['POST'])
def handle_inference():
    """Endpoint to receive files and run the pipeline."""
    start_time = time.time()
    try:
        if 'image' not in request.files:
            return "Missing image file", 400
        
        bbox_model_name = request.form.get('bbox_model_name')
        corner_model_name = request.form.get('corner_model_name')
        sam_model_name = request.form.get('sam_model_name')

        if not all([bbox_model_name, corner_model_name, sam_model_name]):
            return "Missing model name parameters", 400

        image_file = request.files['image']
        
        # Get models from cache or load them
        bbox_model, sam_model, sam_processor, corner_model = get_models_from_cache(
            bbox_model_name, corner_model_name, sam_model_name
        )

        image = Image.open(image_file.stream).convert("RGB")
        results = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "solar_panel", "supercategory": "object"}]}
        img_info, results['annotations'], _ = solarutils.run_inference_pipeline_single(bbox_model, sam_model, sam_processor, corner_model, TRANSFORMS, image, "NA", 1, DEVICE, bbox_threshold=0.7)
        results['images'] = [img_info]
        end_time = time.time()
        duration = end_time - start_time
        print(f"inference done in {duration} seconds")
        return jsonify({'results': results, 'inference_time': duration})

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return f"Internal Server Error: {e}", 500


if __name__ == '__main__':
    print("Starting Flask server...")
    # Ensure model directories exist
    os.makedirs(MODELS_BASE_DIR / 'bboxes', exist_ok=True)
    os.makedirs(MODELS_BASE_DIR / 'corners', exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=False)
