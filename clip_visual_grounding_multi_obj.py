import torch
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
from transformers import CLIPProcessor, CLIPModel

clip_model_path = 'openai/clip-vit-large-patch14'
sam_model_path = # TODO: See README.md to get the download link
image_path = './table.jpg'
text_queries = [
    "carrot in the bowl", 
    "coca cola",
]

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model_path)
    model = CLIPModel.from_pretrained(clip_model_path)
    model.to(device)
    return model, processor, device

clip_model, clip_processor, device = load_clip_model()

def compute_clip_similarity(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    similarity = torch.nn.functional.cosine_similarity(image_features, text_features).cpu().item()
    return similarity

def filter_small_masks(mask, min_area=1000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1
    return filtered_mask

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def crop_masked_region(image, mask):
    mask_np = np.array(mask)
    coords = np.argwhere(mask_np > 0)

    if coords.shape[0] == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    return cropped_image

sam_model = build_efficient_sam(
    encoder_patch_embed_dim=192, 
    encoder_num_heads=3, 
    checkpoint=sam_model_path
).eval().to(device)


image = Image.open(image_path).convert("RGB")
original_size = image.size
image_resized = image.resize((256, 256))
image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(device)

output_dir = "./segmentation_results"
os.makedirs(output_dir, exist_ok=True)

def visualize_sift_keypoints(image, keypoints, save_path):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for (x, y) in keypoints:
        cv2.circle(image_cv, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    cv2.imwrite(save_path, image_cv)

def visualize_mask_on_image(original_img, mask, save_path):
    img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    mask_cv = mask.astype(np.uint8) * 255
    
    colored_mask = np.zeros_like(img_cv)
    colored_mask[:, :, 2] = mask_cv
    
    alpha = 0.5
    overlay = cv2.addWeighted(img_cv, 1.0, colored_mask, alpha, 0)

    cv2.imwrite(save_path, overlay)

def get_sift_keypoints(image, num_points=50):
    image_gray = np.array(image.convert("L"))
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image_gray, None)

    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:num_points]
    return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

# Extract SIFT keypoints
input_points_list = get_sift_keypoints(image_resized, num_points=80)

sift_keypoints_save_path = os.path.join(output_dir, "sift_keypoints.jpg")
visualize_sift_keypoints(image_resized, input_points_list, sift_keypoints_save_path)
print(f"SIFT Keypoints are saved: {sift_keypoints_save_path}")

text_queries_crop = text_queries
text_queries_position = text_queries

all_points = []
all_labels = []
for point in input_points_list:
    # point: (x, y)
    all_points.append([point])  # shape: (1,2)
    all_labels.append([1])      # shape: (1,)

points_tensor = torch.tensor([all_points], dtype=torch.float, device=device)  # shape (1, N, 1, 2)
labels_tensor = torch.tensor([all_labels], dtype=torch.int, device=device)    # shape (1, N, 1)

with torch.no_grad():
    predicted_logits, predicted_iou = sam_model(image_tensor, points_tensor, labels_tensor)
    # predicted_logits shape: [B=1, 1, N, H, W]
    # predicted_iou    shape: [B=1, N]

print("len(input_points_list) = ", len(input_points_list))
print(f"predicted_logits.shape: {predicted_logits.shape}")

predicted_masks = []
for idx in range(len(input_points_list)):
    logits_i = predicted_logits[0, idx, 0, :, :].cpu().numpy()
    mask_i = (logits_i >= 0).astype(np.uint8)

    mask_i = filter_small_masks(mask_i, min_area=20)
    if np.sum(mask_i) == 0:
        predicted_masks.append(None)
        continue

    mask_resized = cv2.resize(mask_i, original_size, interpolation=cv2.INTER_NEAREST)
    predicted_masks.append(mask_resized)

print("len(predicted_masks) = ", len(predicted_masks))

unique_masks = []
iou_threshold = 0.9

for i, mask_i in enumerate(predicted_masks):
    if mask_i is None:
        continue

    duplicate_found = False
    for mask_j in unique_masks:
        iou_val = compute_iou(mask_i, mask_j)
        if iou_val > iou_threshold:
            duplicate_found = True
            break

    if not duplicate_found:
        unique_masks.append(mask_i)

print(f"Number of valid partitions retained after de-duplication: {len(unique_masks)}")

crop_images = []
segmented_images = []
blurred_image = image.filter(ImageFilter.GaussianBlur(10))

for mask_resized in unique_masks:
    mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8)).convert("L")
    segmented_images.append(Image.composite(image, blurred_image, mask_pil))
    cropped_region = crop_masked_region(image, mask_pil)
    crop_images.append(cropped_region)

for idx, segmented_image in enumerate(segmented_images):
    seg_vp_path = os.path.join(output_dir, f"seg_result_vp_{idx}.jpg")
    segmented_image.save(seg_vp_path)
    print(f"Visual Prompting Seg Semantic Segmentation Results for '{idx}' Saved. {seg_vp_path}")

for idx, crop_image in enumerate(crop_images):
    crop_vp_path = os.path.join(output_dir, f"crop_result_vp_{idx}.jpg")
    crop_image.save(crop_vp_path)
    print(f"Visual Prompting Crop Semantic Segmentation Results for '{idx}' Saved. {seg_vp_path}")

time_start = time.time()
top_matches = []

for text_query, text_query_crop in zip(text_queries_position, text_queries_crop):
    best_match = None
    best_similarity = -1
    best_mask = None

    for idx in range(len(segmented_images)):
        similarity_original = compute_clip_similarity(segmented_images[idx], text_query)
        similarity_crop     = compute_clip_similarity(crop_images[idx], text_query_crop)
        final_similarity    = (similarity_original + similarity_crop) / 2
        
        if final_similarity > best_similarity:
            best_similarity = final_similarity
            best_match = segmented_images[idx]
            best_mask = unique_masks[idx]

    print(f"The best similarity of Text query: {text_query}  {best_similarity:.4f}")
    if best_similarity > 0.18:
        top_matches.append((best_match, best_mask, text_query))

time_end = time.time()
print("Computation time =", time_end - time_start)

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 165, 0),
    (128, 0, 128),
    (0, 255, 255),
    (255, 192, 203),
    (165, 42, 42),
    (0, 0, 0),
    (255, 255, 255),
    (192, 192, 192),
    (128, 128, 128),
    (0, 128, 0),
    (0, 0, 128),
    (128, 0, 0),
    (255, 215, 0),
    (75, 0, 130),
    (173, 216, 230),
    (240, 128, 128),
    (152, 251, 152),
    (244, 164, 96),
]
draw = ImageDraw.Draw(image)
for idx, (match, mask, label) in enumerate(top_matches):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = colors[idx % len(colors)]
    for contour in contours:
        points = [tuple(pt[0]) for pt in contour]
        draw.line(points + [points[0]], fill=color, width=3)
    draw.text((10, 30 * idx), label, fill=color, 
              font=ImageFont.truetype("arial.ttf", 40))

best_match_path = os.path.join(output_dir, "best_match_labeled.jpg")
image.save(best_match_path)
print(f"The consolidated labelling results have been saved to: {best_match_path}")