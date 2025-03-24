import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import torchvision.transforms as transforms
import clip
from torchvision import models
from nuimages import NuImages
from Deeplabv3 import network
import os
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

clip_model_path = 'openai/clip-vit-large-patch14'
yolo_model_path = # TODO: See README.md to get the download link
deeplab_model_path = # TODO: See README.md to get the download link
image_path = "road.jpg"
text_query_bbox = "A Jeep with a spare tire mounted on the back."
text_query_blur = "A Jeep with a spare tire mounted on the back."
text_query_custom_blur = "A Jeep with a spare tire mounted on the back."

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model_path)
    model = CLIPModel.from_pretrained(clip_model_path)
    model.to(device)
    return model, processor, device

def compute_clip_similarity(model, processor, device, image_path, text):
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    
    return similarity.item()

def load_deeplab_model(model_path):
    num_classes = 19
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=num_classes, output_stride=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state'])
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for DeepLabV3 segmentation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), np.array(image)

def segment_image(model, image_tensor):
    """Run DeepLabV3 model to get segmentation mask."""
    with torch.no_grad():
        output = model(image_tensor)[0]
    return torch.argmax(output, dim=0).byte().cpu().numpy()

def apply_variable_blur(image, bbox, segmentation_mask, 
                        target_classes=[0, 13, 14, 15],  # [0, 1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18]
                        light_blur_std=20, strong_blur_std=110):
    """Apply variable blur based on segmentation mask and bounding box."""
    image_pil = image.copy()
    strong_blurred = image_pil.filter(ImageFilter.GaussianBlur(strong_blur_std))
    light_blurred = image_pil.filter(ImageFilter.GaussianBlur(light_blur_std))

    bbox_mask = Image.new("L", image_pil.size, 0)
    seg_mask = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(bbox_mask)
    
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=255)

    seg_pixels = np.where(np.isin(segmentation_mask, target_classes))
    for x, y in zip(seg_pixels[1], seg_pixels[0]):
        seg_mask.putpixel((x, y), 255)

    intersection_mask = ImageChops.multiply(bbox_mask, seg_mask)
    seg_only_mask = ImageChops.subtract(seg_mask, bbox_mask)
    background_mask = ImageChops.invert(seg_mask)

    blended = Image.composite(strong_blurred, image_pil, background_mask)
    blended = Image.composite(light_blurred, blended, seg_only_mask)

    return blended

def get_bboxes(nuim, sample_data_token):
    """Retrieve bounding boxes for objects in the sample data."""
    bbox_list = []
    for obj_ann in nuim.object_ann:
        if obj_ann['sample_data_token'] == sample_data_token:
            x1, y1, x2, y2 = obj_ann['bbox']
            bbox_list.append((x1, y1, x2, y2))
    return bbox_list

def load_yolo_model(model_path):
    """Load YOLO model for object detection."""
    model = YOLO(model_path)  # Load YOLO model
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

def get_bboxes_from_yolo(model, image):
    """Run YOLO model on the image and extract bounding boxes."""
    results = model(image)  # Run YOLO on the image
    bbox_list = []

    for result in results:
        for box in result.boxes.xyxy:  # Extract bounding boxes
            x1, y1, x2, y2 = map(int, box.tolist())  # Convert to integer
            bbox_list.append((x1, y1, x2, y2))

    return bbox_list

def draw_bbox_on_image(image, best_bbox, similar_bboxes, output_path):
    """Draw bounding boxes on the original image and save.
    
    - Red: Best match bbox
    - Green: Other highly similar bboxes (within 0.001 of best)
    """
    draw = ImageDraw.Draw(image)

    # Draw similar bboxes (Green)
    for bbox in similar_bboxes:
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="green", width=5)

    # Draw best bbox (Red)
    draw.rectangle([best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3]], outline="red", width=5)

    # Save the image with bounding boxes
    image.save(output_path)

def compute_image_features(model, processor, device, image, bbox, save_path):
    """Compute image features from CLIP using both direct bbox extraction and Gaussian blur."""
    
    # Extract the bbox region from the image
    bbox_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    bbox_image.save(f"{save_path}_bbox.jpg")  # Save bbox image
    
    # Process the bbox image
    inputs_bbox = processor(images=bbox_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features_bbox = model.get_image_features(pixel_values=inputs_bbox["pixel_values"]).squeeze(0)
    
    return image_features_bbox

def apply_custom_blur(image, bbox, segmentation_mask, blur_std=110):
    """Apply custom blur: keep bbox and segmentation=0 areas sharp, blur the rest."""
    image_pil = image.copy()
    blurred = image_pil.filter(ImageFilter.GaussianBlur(blur_std))
    mask = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw the bounding box as a clear region
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=255)
    
    # Add segmentation regions where mask == 0 (background)
    seg_pixels = np.where(segmentation_mask == 0)
    for x, y in zip(seg_pixels[1], seg_pixels[0]):
        mask.putpixel((x, y), 255)
    
    # Composite final image
    final_image = Image.composite(image_pil, blurred, mask)
    return final_image

def compute_combined_image_features(model, processor, device, image, bbox, segmentation_mask, save_path, text_queries):
    """Compute the final image features by using different text queries for each image processing method."""
    
    text_query_bbox, text_query_blur, text_query_custom_blur = text_queries
    
    # 1. BBox image features
    image_features_bbox = compute_image_features(model, processor, device, image, bbox, save_path)
    text_inputs_bbox = processor(text=[text_query_bbox], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features_bbox = model.get_text_features(input_ids=text_inputs_bbox["input_ids"], attention_mask=text_inputs_bbox["attention_mask"]).squeeze(0)
    similarity_bbox = torch.nn.functional.cosine_similarity(image_features_bbox, text_features_bbox, dim=0)
    
    # 2. Blurred image features
    blurred_image = apply_variable_blur(image, bbox, segmentation_mask)
    blurred_image.save(f"{save_path}_blurred.jpg")
    inputs_blur = processor(images=blurred_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features_blur = model.get_image_features(pixel_values=inputs_blur["pixel_values"]).squeeze(0)
    text_inputs_blur = processor(text=[text_query_blur], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features_blur = model.get_text_features(input_ids=text_inputs_blur["input_ids"], attention_mask=text_inputs_blur["attention_mask"]).squeeze(0)
    similarity_blur = torch.nn.functional.cosine_similarity(image_features_blur, text_features_blur, dim=0)
    
    # 3. Custom blur method (bbox + segmentation=0 areas clear, rest blurred)
    custom_blurred_image = apply_custom_blur(image, bbox, segmentation_mask)
    custom_blurred_image.save(f"{save_path}_custom_blurred.jpg")
    inputs_custom_blur = processor(images=custom_blurred_image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features_custom_blur = model.get_image_features(pixel_values=inputs_custom_blur["pixel_values"]).squeeze(0)
    text_inputs_custom_blur = processor(text=[text_query_custom_blur], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features_custom_blur = model.get_text_features(input_ids=text_inputs_custom_blur["input_ids"], attention_mask=text_inputs_custom_blur["attention_mask"]).squeeze(0)
    similarity_custom_blur = torch.nn.functional.cosine_similarity(image_features_custom_blur, text_features_custom_blur, dim=0)
    
    # Compute final features by averaging all three feature vectors
    final_image_features = (image_features_bbox + image_features_blur + image_features_custom_blur) / 3
    
    return final_image_features, similarity_bbox.item(), similarity_blur.item(), similarity_custom_blur.item()

text_queries = (text_query_bbox, text_query_blur, text_query_custom_blur)
image = Image.open(image_path).convert("RGB")

# Load models
yolo_model = load_yolo_model(yolo_model_path)
deeplab_model = load_deeplab_model(deeplab_model_path)
clip_model, clip_processor, device = load_clip_model()

# Get bounding boxes from YOLO
bbox_list = get_bboxes_from_yolo(yolo_model, image)

# Get segmentation mask
image_tensor, _ = preprocess_image(image)
segmentation_mask = segment_image(deeplab_model, image_tensor)

best_match = None
best_score = -1
best_bbox = None
bbox_scores = {}

output_dir = "./segmentation_results"
os.makedirs(output_dir, exist_ok=True)

for i, bbox in enumerate(bbox_list):
    save_path = f"./segmentation_results/bbox_{i}"  # Define save path for images
    
    # Compute combined image features with different text queries
    image_features, similarity_bbox, similarity_blur, similarity_custom_blur = compute_combined_image_features(
        clip_model, clip_processor, device, image, bbox, segmentation_mask, save_path, text_queries
    )
    
    # Compute final similarity by averaging the three similarities
    final_similarity = (similarity_bbox + similarity_blur + similarity_custom_blur) / 3
    print(f"BBox {i}: similarity_bbox = {similarity_bbox}, similarity_blur = {similarity_blur}, similarity_custom_blur = {similarity_custom_blur}, final_similarity = {final_similarity}")
    
    bbox_scores[bbox] = final_similarity
    if final_similarity > best_score:
        best_score = final_similarity
        best_bbox = bbox

# Find all bboxes that have similarity within 0.005 of best_score
similar_bboxes = [bbox for bbox, score in bbox_scores.items() if abs(score - best_score) <= 0.005]

if best_bbox:
    best_result_image_path = "./segmentation_results/best_output.jpg"
    original_image = Image.open(image_path).convert("RGB")
    draw_bbox_on_image(original_image, best_bbox, similar_bboxes, best_result_image_path)
    print(f"Best match saved as {best_result_image_path} with score {best_score}")

