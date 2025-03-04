#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageDraw
import clip
import torch

def convert_box_xywh_to_xyxy(box):
    """
    Convert a bounding box from (x, y, width, height) format to (x1, y1, x2, y2).
    """
    # If 'box' is a tensor, convert it to a NumPy array or list first.
    if hasattr(box, "cpu"):
        box = box.cpu().numpy()
    # Convert coordinates to integers
    x1 = int(round(box[0]))
    y1 = int(round(box[1]))
    x2 = int(round(box[0] + box[2]))
    y2 = int(round(box[1] + box[3]))
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    # If segmentation_mask is a torch tensor, move it to CPU and convert to numpy.
    if isinstance(segmentation_mask, torch.Tensor):
        segmentation_mask = segmentation_mask.cpu().numpy()
    
    # Optionally, ensure the mask is boolean for proper indexing.
    if segmentation_mask.dtype != np.bool_:
        segmentation_mask = segmentation_mask.astype(bool)
    
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    
    # Use the mask to extract the segmented region.
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    
    # Create a black background image.
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    
    # Create a transparency mask for pasting.
    transparency_mask = np.zeros(segmentation_mask.shape, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

@torch.no_grad()
def retriev(elements, search_text, model, preprocess, device):
    """
    Given a list of PIL images and a search text, compute softmaxed CLIP scores.
    """
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute scaled cosine similarity scores and softmax them
    probs = 100.0 * image_features @ text_features.T
    # return probs[:, 0].softmax(dim=0)
    return probs[:, 0] # shape [N]

def get_indices_of_values_above_threshold(values, threshold):
    """
    Return a list of indices for which the score is above the specified threshold.
    """
    return [i for i, v in enumerate(values) if v > threshold]

def load_masks():
    """
    Placeholder for loading or defining your segmentation proposals.
    Replace this function with your own code to load the proposals from SAM.
    
    Expected format:
      A list of dictionaries, where each dictionary has:
         - 'segmentation': a binary (boolean or 0/1) NumPy array.
         - 'bbox': a list or tuple [x, y, width, height].
    """
    raise NotImplementedError("Implement load_masks() to return your SAM proposals")

def add_padding(boxes, padding, img_size=(512, 512)):
    """
    Expands bounding boxes with padding, keeping them within image bounds.
    
    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) with [x_min, y_min, x_max, y_max].
        padding (int): Number of pixels to expand in all directions.
        img_size (tuple): Image dimensions (width, height).
    
    Returns:
        torch.Tensor: Padded bounding boxes.
    """
    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Expand bounding boxes
    x_min = (x_min - padding).clamp(min=0)
    y_min = (y_min - padding).clamp(min=0)
    x_max = (x_max + padding).clamp(max=img_size[0])
    y_max = (y_max + padding).clamp(max=img_size[1])

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def run_extract_category(masks, non_padded_boxes, padding, image, search_text="log", threshold=0.05):
    # Add padding in some way
    boxes = add_padding(non_padded_boxes, padding)

    # Crop regions defined by each segmentation proposal using its bounding box.
    cropped_boxes = []
    for mask, box in zip(masks, boxes):
        bbox_xyxy = convert_box_xywh_to_xyxy(box)
        segmented = segment_image(image, mask)
        cropped = segmented.crop(bbox_xyxy)
        cropped_boxes.append(cropped)

    # Load the CLIP model (choose GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Retrieve and score the cropped proposals with CLIP
    scores = retriev(cropped_boxes, search_text, model, preprocess, device)
    indices = get_indices_of_values_above_threshold(scores, threshold)
    print(f"Indices of values above threshold: {indices} and their respective scores: {scores[indices]}")

    return indices, boxes
