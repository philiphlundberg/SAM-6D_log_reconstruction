# set the paths
export CAD_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/cyl.ply   # path to a given cad model(mm)
export RGB_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/multiple_logs.png           # path to a given RGB image
export DEPTH_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/multiple_logs_d.png       # path to a given depth map(mm)
export CAMERA_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/camera_logs_perspective.json    # path to given camera intrinsics
export OUTPUT_DIR=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/outputs         # path to a pre-defined file for saving results
export BLENDER_PATH=/home/philiph/Blender/blender-3.3.1-linux-x64


# Render CAD templates
cd Render
# blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --custom-blender-path $BLENDER_PATH #--colorize True 

# Run instance segmentation model
export SEGMENTOR_MODEL=sam
export STABILITY_SCORE_THRESH=0.9
export SEARCH_TEXT="rock"

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --stability_score_thresh $STABILITY_SCORE_THRESH --search_text $SEARCH_TEXT   

echo "Instance Segmentation Done"

# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
# python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

