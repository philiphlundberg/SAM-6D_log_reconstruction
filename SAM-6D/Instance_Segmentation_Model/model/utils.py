import torch
import numpy as np
import torchvision
from torchvision.ops.boxes import batched_nms, box_area
import logging
from utils.inout import save_json, load_json, save_npz
from utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask
import time
from PIL import Image

lmo_object_ids = np.array(
    [
        1,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]
)  # object ID of occlusionLINEMOD is different


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data) -> None:
        if isinstance(data, str):
            data = self.load_from_file(data)
        for key, value in data.items():
            setattr(self, key, value)
        self.keys = list(data.keys())
        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray):
                self.to_torch()
            self.boxes = self.boxes.long()

    def remove_very_small_detections(self, config):
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_idxs = torch.logical_and(
            box_areas > config.min_box_size**2, mask_areas > config.min_mask_size
        )
        # logging.info(f"Removing {len(keep_idxs) - keep_idxs.sum()} detections")
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms_per_object_id(self, nms_thresh=0.5):
        keep_idxs = BatchedData(None)
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)
        for object_id in torch.unique(self.object_ids):
            idx = self.object_ids == object_id
            idx_object_id = all_indexes[idx]
            keep_idx = torchvision.ops.nms(
                self.boxes[idx].float(), self.scores[idx].float(), nms_thresh
            )
            keep_idxs.cat(idx_object_id[keep_idx])
        keep_idxs = keep_idxs.data
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms(self, nms_thresh=0.5):
        keep_idx = torchvision.ops.nms(
            self.boxes.float(), self.scores.float(), nms_thresh
        )
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])
    
    def apply_containment_suppression_for_id_2(self, tolerance=5):
        """
        NOTE: Object ID before this is 1, really dumb. (2 = 1)
        Suppresses any box (with object_id == 2) whose corners are entirely
        within another (larger) box of the same object_id (== 2).
        
        Leaves all other boxes (object_id != 2) untouched.
        Updates all attributes in `self.keys` accordingly.
        """
        # We'll create a keep mask for all boxes, default True
        num_boxes = len(self.boxes)
        keep = [True] * num_boxes
        
        # Retrieve the object IDs 
        object_ids = self.object_ids  # shape: (num_boxes,)
        
        for i in range(num_boxes):
            # Only check containment if the current box is object_id == 2
            if object_ids[i] != 1:
                continue  # skip if not object_id == 2
            
            if not keep[i]:
                # Already marked for removal
                continue
            
            x1_i, y1_i, x2_i, y2_i = self.boxes[i]
            
            # Compare with all other boxes that have object_id == 2
            for j in range(num_boxes):
                if j == i or object_ids[j] != 1:
                    continue
                
                x1_j, y1_j, x2_j, y2_j = self.boxes[j]
                
                # Check if box i is fully inside box j
                fully_contained = (
                    x1_j - tolerance <= x1_i and
                    y1_j - tolerance <= y1_i and
                    x2_j + tolerance >= x2_i and
                    y2_j + tolerance>= y2_i
                )
                
                if fully_contained:
                    # Mark box i for removal (it's contained by box j)
                    keep[i] = False
                    break
        
        # Gather indices of boxes to keep
        keep_idx = [idx for idx, val in enumerate(keep) if val]

        # Filter all attributes in `self.keys`
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])

        print("Done with containment suppression for object_id == 2")

    def apply_mask_dot_nms_category1(self, mask_threshold=0.5):
        """
        For objects with category 1, compares each pair of binary masks using a dot product
        measure. For each pair (i, j) among masks with object_id == 1:
        
        - Compute ratio1 = (mask_i * mask_j).sum() / (mask_j.sum() + eps)
        - Compute ratio2 = (mask_i * mask_j).sum() / (mask_i.sum() + eps)
        
        If ratio1 equals 1, then every pixel of mask_j is also set in mask_i, meaning mask_j
        is completely contained in mask_i. If ratio2 equals 1, then mask_i is fully contained in
        mask_j. In these cases, the mask with the lower score (or the smaller area) is suppressed.
        
        All attributes in self.keys are updated accordingly.
        """
        import torch
        eps = 1e-6  # small constant to avoid division by zero
        
        # Get indices for detections with category 1
        cat1_idx = (self.object_ids == 1).nonzero(as_tuple=True)[0]
        
        # Binarize the masks using a threshold
        binary_masks = (self.masks > mask_threshold).float()  # shape: [N, H, W]
        
        num_dets = len(self.masks)
        # Create a boolean mask for all detections (True = keep)
        keep = torch.ones(num_dets, dtype=torch.bool, device=self.masks.device)
        
        # Compare every pair among the category 1 detections
        for i in range(len(cat1_idx)):
            idx_i = cat1_idx[i].item()
            mask_i = binary_masks[idx_i]
            sum_i = mask_i.sum() + eps  # total "on" pixels in mask_i
            
            for j in range(i + 1, len(cat1_idx)):
                idx_j = cat1_idx[j].item()
                mask_j = binary_masks[idx_j]
                sum_j = mask_j.sum() + eps  # total "on" pixels in mask_j
                
                
                # Compute the intersection between the two masks
                intersection = (mask_i * mask_j).sum()
                # print("intersection: ", intersection)
                # Ratio: how much of mask_j is covered by mask_i
                # Det här är a*b/sum(b)
                ratio1 = intersection / sum_j
                # print("ratio1: ", ratio1)
                # Ratio: how much of mask_i is covered by mask_j
                # Det här är a*b/sum(a)
                ratio2 = intersection / sum_i  
                # print("ratio2: ", ratio2)
                
                # If mask_i completely covers mask_j
                if ratio1 >= 0.97:
                    if sum_i >= sum_j:
                        keep[idx_j] = False  # suppress smaller mask_j
                    else:
                        keep[idx_i] = False  # suppress smaller mask_i

                # If mask_j completely covers mask_i
                if ratio2 >= 0.97:
                    if sum_j >= sum_i:
                        keep[idx_i] = False
                    else:
                        keep[idx_j] = False

                # If mask_i completely covers mask_j, then ratio1 should be close to 1
                # if ratio1 >= 0.97:
                #     # Option: keep the one with the higher score
                #     if self.scores[idx_i] >= self.scores[idx_j]:
                #         keep[idx_j] = False  # suppress mask_j
                #     else:
                #         keep[idx_i] = False  # suppress mask_i
                
                # # Similarly, if mask_j completely covers mask_i, ratio2 will be 1
                # if ratio2 >= 0.97:
                #     if self.scores[idx_j] >= self.scores[idx_i]:
                #         keep[idx_i] = False  # suppress mask_i
                #     else:
                #         keep[idx_j] = False  # suppress mask_j
                        
        # Update all attributes in self.keys to include only the kept detections
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep])

    def apply_mask_area_filter(self, min_area=2000, mask_threshold=0.5):
        """
        Removes detections whose binary mask area is below a specified threshold.

        Parameters:
        - min_area (int): Minimum number of pixels required for a mask to be kept.
        - mask_threshold (float): Threshold to binarize the masks.
        """

        # Binarize the masks using the specified threshold
        binary_masks = (self.masks > mask_threshold).float()  # Shape: [N, H, W]

        # Compute the area (number of 'on' pixels) for each mask
        mask_areas = binary_masks.view(binary_masks.shape[0], -1).sum(dim=1)

        # Create a boolean mask indicating which detections to keep
        keep = mask_areas >= min_area

        # Update all relevant attributes to include only the kept detections
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep])



    def check_object_ids(self):
        # Print each id and the corresponding bounding box
        for i in range(len(self.object_ids)):
            print(f"Object ID: {self.object_ids[i]}")
            print(f"Bounding Box: {self.boxes[i]}")


    def add_attribute(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)
        assert (
            mask_size == box_size == score_size == object_id_size
        ), f"Size mismatch {mask_size} {box_size} {score_size} {object_id_size}"

    def to_numpy(self):
        for key in self.keys:
            setattr(self, key, getattr(self, key).cpu().numpy())

    def to_torch(self):
        for key in self.keys:
            a = getattr(self, key)
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(
        self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """

        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
            "segmentation": self.masks,
        }
        save_npz(file_path, results)
        if return_results:
            return results

    def load_from_file(self, file_path):
        data = np.load(file_path)
        masks = data["segmentation"]
        boxes = xywh_to_xyxy(np.array(data["bbox"]))
        data = {
            "object_ids": data["category_id"] - 1,
            "bbox": boxes,
            "scores": data["score"],
            "masks": masks,
        }
        logging.info(f"Loaded {file_path}")
        return data

    def filter(self, idxs):
        for key in self.keys:
            setattr(self, key, getattr(self, key)[idxs])

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())


def convert_npz_to_json(idx, list_npz_paths):
    npz_path = list_npz_paths[idx]
    detections = np.load(npz_path)
    results = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": float(detections["score"][idx_det]),
            "time": float(detections["time"]),
            "segmentation": mask_to_rle(
                force_binary_mask(detections["segmentation"][idx_det])
            ),
        }
        results.append(result)
    return results
