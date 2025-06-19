#!/bin/bash
# set the paths
export CAD_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/cyl2.ply   # path to a given cad model(mm)
export RGB_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1/snapshot_rgb.png         # path to a given RGB image
export DEPTH_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1/snapshot_depth.png       # path to a given depth map(mm)
export CAMERA_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/camera_logs.json    # path to given camera intrinsics
export OUTPUT_DIR=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction         # path to a pre-defined file for saving results
export BLENDER_PATH=/home/philiph/Blender/blender-3.3.1-linux-x64
export HF_OUTPUT_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1/heightfields/heightfield.npz

source /home/philiph/miniconda3/etc/profile.d/conda.sh
# conda init
conda activate sam6d


export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/bin/virtualenv
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
workon autoscene

####### CREATING SIMULATION IMAGES #######
# python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:3 \
# --controller DoNothing:120 LoadLogsFromNPZ DoNothing:120 LogStateRecorder

# python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:3 \
# --controller DoNothing:120 DropLogs DoNothing:300 LogStateRecorder TakeSnapshot

# python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:4 \
# --controller DropAndEmbedLogs LogStateRecorder TakeSnapshot
##########################################



deactivate
cd SAM-6D/SAM-6D



########### RUNNING SAM6D ###########
### Render CAD templates
cd Render
# blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --custom-blender-path $BLENDER_PATH #--colorize True 
###

export SEGMENTOR_MODEL=sam
export STABILITY_SCORE_THRESH=0.97
export SEARCH_TEXT="A cut wooden log."


### Run instance segmentation model
cd ../Instance_Segmentation_Model
# python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH \
# --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --stability_score_thresh $STABILITY_SCORE_THRESH --search_text "$SEARCH_TEXT"   
# ###


cd ../
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json
export DET_SCORE_THRESH=0.32


### Run pose estimation model
cd Pose_Estimation_Model
# python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH \
# --cam_path $CAMERA_PATH --seg_path $SEG_PATH --det_score_thresh $DET_SCORE_THRESH
# ###

### Options for clip:
#  "A boulder or stone in a natural outdoor setting."
#  "A cut wooden log, possibly with parts obscured by other objects."
##########################################






cd ../../../
workon autoscene

# ###### RUNNING AGX-PIPELINE #######
# ## Create a HeightField from the depth map
# python generate_heightfield.py --depth_path "$DEPTH_PATH" --output "$HF_OUTPUT_PATH" \
# --det_dir "$OUTPUT_DIR/sam6d_results/detection_ism.json" --downsampling 4 --camera_yaml "settings/settings.yml"
# ##
#####################################

### Visualization on the original heightfield
# python run.py --environment logpile --settings-file Test2/settings_default.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:120 LoadLogsFromJSON DoNothing:inf
###

### Visualization before
python run.py --environment logpile --settings-file Test2/settings_before.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:120 LoadLogsFromJSON DoNothing:120
# ###

# # ### Store pose with just initial guess
python run.py --environment logpile --settings-file Test2/settings_intermediate.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:120 LoadLogsFromJSON DoNothing:300


# # # ## Optimization
python run.py --environment logpile --settings-file Test2/settings_before.yml --agxOnly --spawner TreeLog:3 \
--controller AddObserver HeightfieldOptimizer:'sam6d_results/detection_pem.json' 
##


### Visualization after
python run.py --environment logpile --settings-file Test2/settings_after.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:120 LoadLogsFromJSON DoNothing:300

python run.py --environment logpile --agxOnly --settings-file Test2/eval_gt_init.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
python run.py --environment logpile --agxOnly --settings-file Test2/eval_gt_opt.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
python run.py --environment logpile --agxOnly --settings-file Test2/eval_gt_sam6d.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
python run.py --environment logpile --agxOnly --settings-file Test2/eval_sam6d_init.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
python run.py --environment logpile --agxOnly --settings-file Test2/eval_sam6d_opt.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
python run.py --environment logpile --agxOnly --settings-file Test2/eval_init_opt.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator

####################################

deactivate
cd SAM-6D/SAM-6D
