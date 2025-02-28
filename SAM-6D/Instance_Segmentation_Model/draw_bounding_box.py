import cv2
import numpy as np

def draw_bounding_boxes(image_path, bboxes, output_path=None):
    """
    Loads an image and visualizes bounding boxes.
    
    Parameters:
        image_path (str): Path to the input image.
        bboxes (torch.Tensor or numpy.ndarray): Bounding boxes in [x1, y1, x2, y2] format.
            Example tensor:
                tensor([[  0,   0, 511, 511],
                        [164, 114, 190, 329],
                        [124, 231, 168, 268],
                        [138, 113, 191, 329],
                        [124, 231, 304, 362]], device='cuda:0')
        output_path (str, optional): Path to save the output image. If None, displays the image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Convert BGR to RGB for proper visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert tensor to numpy array if necessary
    if hasattr(bboxes, "cpu"):
        bboxes = bboxes.cpu().numpy()
    
    # Draw each bounding box
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue box
    
    # Display or save the image
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imshow("Bounding Boxes", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    image_path = "example.jpg"  # Change to your image file
    # For demonstration purposes, we'll create a sample numpy array resembling a tensor
    import numpy as np
    sample_boxes = np.array([
        [0, 0, 511, 511],
        [164, 114, 190, 329],
        [124, 231, 168, 268],
        [138, 113, 191, 329],
        [124, 231, 304, 362]
    ])
    draw_bounding_boxes(image_path, sample_boxes)
