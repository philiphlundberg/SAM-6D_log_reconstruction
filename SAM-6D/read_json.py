import json
import numpy as np
from Instance_Segmentation_Model.draw_bounding_box import draw_bounding_boxes

# path to detection_ISM json
path = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/outputs/sam6d_results/detection_ism.json"
# path to rgb image
img_path = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/logs3.png"
# output path
out_path = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/bboxes_from_json.png"

boxes = []

with open(path, 'r') as file:
    data = json.load(file)
    # Print the bounding boxes for each element that has category id 2:
    for item in data:
        if item["category_id"] != 2:
            continue
        else:
            boxes.append(item["bbox"])
boxes_np = np.asarray(boxes)
draw_bounding_boxes(img_path, boxes, out_path, format="xywh")

