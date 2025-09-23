import numpy as np
import os
import cv2

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

# def calculate_2d_projections_ortho(coords_3d, K, y_down=True, round_to_int=True):
#     """
#     coords_3d: [3, N] camera-frame points in meters (X,Y,Z)
#     K: [[sx,0,cx],[0,sy,cy],[0,0,1]] with sx,sy in px/m
#     y_down: if True, image v increases downward (OpenCV). If your camera Y points up,
#             set y_down=False to flip the sign in v.
#     """
#     sx, sy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
#     X, Y = coords_3d[0, :], coords_3d[1, :]

#     u = sx * X + cx
#     v = sy * Y + cy if y_down else (cy - sy * Y)

#     uv = np.stack([u, v], axis=1)           # [N,2]
#     if round_to_int:
#         uv = np.rint(uv).astype(np.int32)   # round before int to avoid shrink
#     return uv

def project_ortho(coords_3d, K, H=512, W=512, y_down=True, flip_y=True):
    """
    Orthographic projection from 3D camera-frame coords to 2D image pixels.

    Args:
        coords_3d : np.ndarray, shape (3,N)  -> [X,Y,Z] in meters
        K         : 3x3 intrinsic matrix with [ [sx,0,cx],[0,sy,cy],[0,0,1] ]
        H, W      : image height, width in pixels
        y_down    : if True, assumes image v increases downward (OpenCV convention).
        flip_y    : if True, applies vertical flip because VirtualCameraSensor.image
                    uses np.flip(img,0). Set False if you remove that flip.

    Returns:
        uv_int    : np.ndarray, shape (M,2), pixel coords as integers, only in-bounds
    """
    sx, sy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X, Y = coords_3d[0], coords_3d[1]

    # forward ortho projection
    u = sx * X + cx
    v = sy * Y + cy if y_down else (cy - sy * Y)

    uv = np.stack([u, v], axis=1)           # (N,2)

    # flip vertically if the saved image was flipped in the camera
    if flip_y:
        uv[:,1] = H - 1 - uv[:,1]

    # keep only finite & in-bounds points
    mask = np.isfinite(uv).all(axis=1) & \
           (uv[:,0] >= 0) & (uv[:,0] < W) & \
           (uv[:,1] >= 0) & (uv[:,1] < H)
    uv = uv[mask]

    return np.rint(uv).astype(np.int32)     # round then cast


def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def project_ortho_bbox(coords_3d, K, H, W, flip_y=False):
    sx, sy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X, Y = coords_3d[0], coords_3d[1]

    u = sx * X + cx
    v = sy * Y + cy
    if flip_y:
        v = H - 1 - v

    uv = np.stack([u, v], axis=1)
    return np.rint(uv).astype(np.int32)   # [8,2], even if outside


def draw_3d_bbox(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, color=(0, 255, 0)):
    K = intrinsics[0]
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    # 3d bbox
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    # 3d point
    choose = np.random.choice(np.arange(len(model_points)), 200) # 512
    pts_3d = model_points[choose].T

    for ind in range(num_pred_instances):
        # draw 3d bounding box

        H, W = image.shape[:2]

        transformed_bbox_3d = pred_rots[ind] @ bbox_3d + pred_trans[ind][:,None]
        projected_bbox = project_ortho_bbox(transformed_bbox_3d, K, H, W, flip_y=False)
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color=(255,0,0), size=1)

        transformed_pts_3d = pred_rots[ind] @ pts_3d + pred_trans[ind][:,None]
        projected_pts = project_ortho(transformed_pts_3d, K, H, W, flip_y=False)  # this one can filter
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, (0,255,0), size=1)



        # transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
        # projected_bbox = project_ortho(transformed_bbox_3d, intrinsics[ind])
        # draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color=(255,0,0), size=1)
        # # draw point cloud
        # transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
        # projected_pts = project_ortho(transformed_pts_3d, intrinsics[ind])
        # draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

    return draw_image_bbox

if __name__ == "__main__":
    # quick sanity check
    H, W = 512, 512
    K = np.array([[64,0,256],
                  [0,64,256],
                  [0, 0,  1]], np.float32)

    # two 3D points, 1 m apart in X
    coords_3d = np.array([[0.0, 1.0],   # X
                          [0.0, 0.0],   # Y
                          [5.0, 5.0]])  # Z (ignored by ortho)

    uv = project_ortho(coords_3d, K, H, W)
    print("Projected pixel coords:", uv)
    print("Δu (should be 64 px):", uv[1,0] - uv[0,0])

