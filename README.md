# Going Off-Grid: A Computer Vision Approach for Grid Integration and Reconstruction in Post-Conflict Syria

[Erdős Institute](https://www.erdosinstitute.org/) [Deep Learning Boot Camp](https://www.erdosinstitute.org/programs/summer-2025/deep-learning-boot-camp), Summer 2025.

![One of our aerial drone photographs](/main_ipynb_images/drone_7.jpg)

## Team Members:
- [Al Baraa Abd Aldaim](https://www.linkedin.com/in/baraa-abd/)
- [Nicholas Geiser](https://www.linkedin.com/in/ngeiser/)
- [Suman Bhandari](https://www.linkedin.com/in/suman-bhandari/)

# Project Description

This project introduces a novel workflow for identifying solar panel assemblies in the challenging, post-war urban landscapes of Syria. Decentralized solar electricity production has become common in Syria due to unreliability and inconsistent delivery from the national electric grid. Our primary goal was to develop a deep learning model capable of not only detecting panels but also performing instance segmentation to estimate their area and keypoint regression to determine each instance's installation angle (i.e., cardinal direction). Both of these outputs are important for accurately estimating energy production from residential solar, which is vital for grid integration and reconstruction efforts.

# Models
Our approach implements a 3 stage workflow, where we use Faster R-CNN for object detection, SlimSAM for segmentation, and a custom architecture with a Resnet50 with FPN backbone (shared with the Faster R-CNN model) for keypoint regression.


# Notebooks
- main.ipynb notebook  -- offers a thorough overview of our workflow and results.
- active_learning.ipynb notebook -- contains our active learning workflow.
- multistage_multitask_training.ipynb notebook -- contains our multistage training routine.

Note that a lot of our core functionality is in the solarutils.py (datasets, models, inference, and helper functions) and solareval.py (evaluation) modules. 

# Dependencies
Running the code requires the following Python packages: 
```
torch, torchvision, torchmetrics, transformers, PIL, numpy, pandas, timm, matplotlib, tqdm
```
# Data
Our custom dataset is publically available for free on this repository under a [Creative Commons (CC) BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). 

It is arranged as follows in the data folder (see main.ipynb for more details): 
  - images.zip and annotations.json: Contains 1544 image patches (224x224) with annotations in COCO JSON format.
  - test_images.zip and test_annotations.json: Contains 109 image patches (224x224), sampled from a source image reserved for test purposes, with annotations in COCO JSON format. 
