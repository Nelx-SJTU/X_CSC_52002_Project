# Two-Stage Semantic-Aware Visual Grounding

This repository contains the official implementation of the article:

> **Two-Stage Semantic-Aware Visual Prompting for Object Grounding in Robotics**  
> *Neng Xu, Shikun Wei*  


We propose a lightweight and modular two-stage object grounding framework using visual prompting and CLIP-based cross-modal similarity, optimized for real-time robotic applications.

---

## Overview of the Two-Stage Framework

### Stage 1: Semantic-Aware Candidate Extraction
We leverage fast and lightweight models to extract potential object candidates:
- **YOLOv8-nano**: for bounding box detection.
- **DeepLabV3+ (MobileNet backbone)**: for semantic segmentation.
- **EfficientSAM**: for fine-grained visual prompting masks.

This step filters noisy regions and reduces grounding complexity in cluttered scenes.

### Stage 2: Visual Prompting and Cross-modal Matching
Each candidate region is enhanced using visual prompting techniques:
- **Region cropping**
- **Multi-level Gaussian blur**
- **Custom blur with segmentation guidance**

These modified regions are encoded by **CLIP**, and their similarity to the text query is computed to determine the best match.

---

## File Descriptions

### [`clip_visual_grounding_multi_obj.py`](./clip_visual_grounding_multi_obj.py)
Implements **visual prompting with EfficientSAM** for **multi-object grounding** in a tabletop setting (e.g., food items, bottles).

#### Key Features
- Uses **SIFT keypoints** for point-based prompting.
- Applies **EfficientSAM (ViT-T)** for segmentation.
- CLIP ranks regions based on text-image similarity.
- Outputs segmented, cropped, and labeled results.

#### Example Use Case
```python
text_queries = [
    "carrot in the bowl",
    "coca cola",
]
```

### [`clip_visual_grounding_yolo.py`](./clip_visual_grounding_yolo.py)
Implements **visual grounding in autonomous driving** scenarios using **YOLO + DeepLab + CLIP**.

#### Key Features
- Detects object candidates via **YOLOv8**.
- Applies **DeepLabV3+** to mask and guide blur.
- Uses **three visual prompt views**:
  - Bbox crop
  - Full image with variable blur
  - Custom blur (background + segmentation filter)
- Final image-text similarity is computed by **averaging scores** across all views.

#### Example Query
```python
text_query = "A Jeep with a spare tire mounted on the back."
```

---

## Requirements

Install dependencies:
```bash
pip install torch torchvision transformers pillow opencv-python matplotlib ultralytics
```

You may also need:
```bash
pip install git+https://github.com/openai/CLIP.git
```

---

## Pretrained Models

| Model | Description | Download |
|-------|-------------|----------|
| `efficient_sam_vitt.pt` | Lightweight ViT-T backbone for EfficientSAM | [Download](https://github.com/yformer/EfficientSAM/blob/main/weights/efficient_sam_vitt.pt) |
| `yolo11n.pt` | YOLOv8-nano object detector (custom trained) | [Download](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) |
| `best_deeplabv3plus_mobilenet_cityscapes_os16.pth` | DeepLabV3+ for semantic segmentation | [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) |
| `clip-vit-large-patch14` | CLIP model for image-text alignment | Use HuggingFace: `from_pretrained("openai/clip-vit-large-patch14")` |

---

## Running the Code

### Multi-Object Grounding (EfficientSAM)
```bash
python clip_visual_grounding_multi_obj.py
```

### Driving Scene Grounding (YOLO + DeepLab)
```bash
python clip_visual_grounding_yolo.py
```

Results are saved in the `./segmentation_results` directory.

---

## Benchmark Results

We evaluate our **Two-Stage Semantic-Aware Visual Prompting** framework on three widely used referring expression datasets: **RefCOCO**, **RefCOCO+**, and **RefCOCOg**. These datasets test the model's ability to localize objects from natural language descriptions in various levels of linguistic complexity.

A prediction is considered correct if its Intersection-over-Union (IoU) with the ground truth is greater than 0.5. Below is a comparison with state-of-the-art baselines.

### 🔬 Accuracy (%) on Visual Grounding Benchmarks

| **Method** | **Visual Prompt** | RefCOCO (val) | testA | testB | RefCOCO+ (val) | testA | testB | RefCOCOg (val) | test |
|------------|-------------------|---------------|--------|--------|------------------|--------|--------|------------------|-------|
| CPT-adapted | B2 | 23.2 | 21.4 | 27.0 | 23.9 | 21.6 | 25.9 | 22.3 | 23.7 |
| CPT-adapted | P \| B2 | 40.1 | 40.0 | 44.0 | 40.1 | 41.8 | 41.1 | 51.3 | 51.9 |
| ReCLIP | P \| B4 | 45.8 | 46.0 | 47.1 | 40.0 | 50.1 | 45.1 | 59.3 | 59.0 |
| RedCircle | P \| C1 | 43.9 | 46.2 | 44.3 | 47.9 | 43.1 | 47.3 | 57.3 | 56.3 |
| Hierarchical Semantic Model | P \| D4 | 46.2 | 48.2 | 45.7 | 48.3 | 51.9 | 49.4 | 59.0 | 58.6 |
| **FGVP** | **P \| D4** | **52.0** | **55.9** | **48.8** | **50.4** | **60.4** | **46.7** | **62.1** | **61.9** |

> 🔹 **Visual Prompt Legend**:  
> - **P** = Region Crop  
> - **B2** = Colorful Box  
> - **B4** = Blur Reverse  
> - **C1** = Circle Prompt  
> - **D4** = Blur with Segmentation Mask

---

## Applications

- **Autonomous Driving**  
  Enables robust car localization using shape- and position-aware queries.

- **Instruction-Guided Manipulation**  
  Grounds object references like “the green bottle” or “carrot in the bowl” in real time.

---

# Re-build the result table

This section provides an additional fine-grained visual perception (FGVP) evaluation module with multi-layer reasoning, which can complement the semantic-aware grounding framework.

## Overview

This module extends the [FGVP](https://github.com/ylingfeng/FGVP) framework by incorporating semantic segmentation features and multi-layer reasoning strategies to improve classification performance in fine-grained visual tasks. The core logic is implemented in `executor.py`.

## Setup Instructions

### 1. Clone the Semantic Segmentation Repository

This module uses DeepLabV3+ for semantic guidance. Clone the required repository:

```bash
git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git
```

### 2. Download Pretrained Segmentation Model

Download the pretrained DeepLabV3+ model using the link mentioned above.

### 3. Install Dependencies

The environment setup and dataset preparation follow the same instructions as provided in the original [FGVP repository](https://github.com/ylingfeng/FGVP).

## Running the Experiment

Run the following command to evaluate the baseline and our proposed method:

```bash
python run_experiment.py
```

## Acknowledgements

This module builds on:

- [FGVP](https://github.com/ylingfeng/FGVP)
- [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)


