import cv2
import numpy as np

# def draw_bounding_boxes(image_path, bboxes, output_path=None, format="xyxy", alpha=0.6):
#     """
#     Loads an image and visualizes bounding boxes.
    
#     Parameters:
#         image_path (str): Path to the input image.
#         bboxes (torch.Tensor or numpy.ndarray): Bounding boxes in [x1, y1, x2, y2] format.
#             Example tensor:
#                 tensor([[  0,   0, 511, 511],
#                         [164, 114, 190, 329],
#                         [124, 231, 168, 268],
#                         [138, 113, 191, 329],
#                         [124, 231, 304, 362]], device='cuda:0')
#         output_path (str, optional): Path to save the output image. If None, displays the image.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Could not load image.")
#         return
    
#     # Convert BGR to RGB for proper visualization
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Convert tensor to numpy array if necessary
#     if hasattr(bboxes, "cpu"):
#         bboxes = bboxes.cpu().numpy()

#     # Make a copy for overlay
#     overlay = image.copy()

#     # Draw each bounding box
#     for bbox in bboxes:
#         if format == "xywh":
#             x1, y1, w, h = bbox
#             x2 = x1 + w
#             y2 = y1 + h
#         elif format == "xyxy":
#             x1, y1, x2, y2 = bbox
#         color = tuple(np.random.randint(0, 255, size=3).tolist())


#         cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

#     image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) 
    
#     # Display or save the image
#     if output_path:
#         cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#     else:
#         cv2.imshow("Bounding Boxes", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


def draw_bounding_boxes(image_path, bboxes, output_path=None, format="xyxy",
                        alpha=0.75, thickness=2, seed=None):
    """
    Draw semi-transparent rectangle edges with proper overlap blending.
    Overlapping edges become more opaque and mix colors.

    alpha: per-box opacity contribution in [0,1]
    thickness: line thickness in pixels
    """
    # Load
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("Error: Could not load image.")
        return

    # Convert to RGB float [0,1]
    base = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Convert tensor to numpy array if necessary
    if hasattr(bboxes, "cpu"):
        bboxes = bboxes.cpu().numpy()
    bboxes = np.asarray(bboxes)

    # Accumulators (premultiplied alpha)
    H, W = base.shape[:2]
    acc_rgb = np.zeros((H, W, 3), dtype=np.float32)
    acc_a   = np.zeros((H, W), dtype=np.float32)

    rng = np.random.default_rng(seed)

    for bbox in bboxes:
        if format == "xywh":
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        else:
            x1, y1, x2, y2 = bbox

        # Clamp to image bounds just in case
        x1, y1 = int(max(0, min(W-1, x1))), int(max(0, min(H-1, y1)))
        x2, y2 = int(max(0, min(W-1, x2))), int(max(0, min(H-1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        # Random color per box (0..1). Swap for deterministic if you want.
        color = rng.random(3, dtype=np.float32)

        # Draw lines for this box on a mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness, lineType=cv2.LINE_AA)

        # Per-pixel contribution (premultiplied)
        a = (mask.astype(np.float32) / 255.0) * float(alpha)
        if a.max() == 0:
            continue
        acc_rgb += (a[..., None] * color[None, None, :])
        acc_a   += a

    # Cap alpha so base image is still visible where many lines overlap
    acc_a = np.clip(acc_a, 0.0, 0.95)

    # Compute resulting edge color (un-premultiply) where alpha > 0
    result = base.copy()
    nonzero = acc_a > 0
    if np.any(nonzero):
        edge_rgb = np.zeros_like(acc_rgb)
        edge_rgb[nonzero] = acc_rgb[nonzero] / acc_a[nonzero, None]
        # Composite over base: out = base*(1-a) + edge*a
        result[nonzero] = base[nonzero] * (1.0 - acc_a[nonzero, None]) + edge_rgb[nonzero] * acc_a[nonzero, None]

    out_bgr = cv2.cvtColor((result * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if output_path:
        cv2.imwrite(output_path, out_bgr)
    else:
        cv2.imshow("Bounding Boxes", out_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    image_path = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/logs3.png"  # Change to your image file
    # For demonstration purposes, we'll create a sample numpy array resembling a tensor
    import numpy as np
    sample_boxes = np.array([
        [170, 178, 177, 138],
        [118, 311, 184, 125],
        [359, 132, 39, 214],
        [257, 178, 90, 70],
        [170, 252, 83, 64],
        [100, 387, 36, 17]
    ])
    draw_bounding_boxes(image_path, sample_boxes, output_path="/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/res.jpg")