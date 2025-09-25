#!/bin/bash
# set the paths
# export CAD_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/cyl2.ply   # path to a given cad model(mm)
# export RGB_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1/snapshot_rgb.png         # path to a given RGB image
# export DEPTH_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1/snapshot_depth.png       # path to a given depth map(mm)
# export CAMERA_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/camera_logs.json    # path to given camera intrinsics
# export OUTPUT_DIR=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction         # path to a pre-defined file for saving results
# export BLENDER_PATH=/home/philiph/Blender/blender-3.3.1-linux-x64
# export HF_OUTPUT_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1/heightfields/heightfield.npz

source /home/philiph/miniconda3/etc/profile.d/conda.sh
# conda init
conda activate sam6d


export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/bin/virtualenv
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
workon autoscene

####### CREATING SIMULATION IMAGES #######
# python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:10 DropLogs DoNothing:inf TakeSnapshot

python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:3 \
--controller DoNothing:20 TakeSnapshot DoNothing:60

# python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:4 \
# --controller DropAndEmbedLogs LogStateRecorder TakeSnapshot
##########################################

deactivate

########### RUNNING SAM6D ###########
### Render CAD templates
cd SAM-6D/SAM-6D/Render
# blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --custom-blender-path $BLENDER_PATH #--colorize True 
###

### Run instance segmentation model
cd ../Instance_Segmentation_Model
python run_inference_custom.py --settings_file ../../../settings/default_settings.yml --section SAM6DInference
###

### Run pose estimation model
cd ../Pose_Estimation_Model
python run_inference_custom.py --settings_file ../../../settings/default_settings.yml --section SAM6DInference
###



workon autoscene




cd ../../../

# ###### RUNNING AGX-PIPELINE #######
# ## Create a HeightField from the depth map
python utils/generate_heightfield.py --settings_file settings/default_settings.yml --section TakeSnapshot
# python utils/generate_heightfield.py --depth_path "$DEPTH_PATH" --output "$HF_OUTPUT_PATH" \
# --det_dir "$OUTPUT_DIR/sam6d_results/detection_ism.json" --downsampling 4 --camera_yaml "settings/default_settings.yml"
# ##


### Visualization before
# Static (Maybe not necessary)
python run.py --environment logpile --settings-file Test2/settings_view_sam6d_static.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:120 LogVisualizer DoNothing:200
###
# Dynamic (Contains all needed information)
python run.py --environment logpile --settings-file Test2/settings_view_sam6d_dynamic.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:120 LogVisualizer DoNothing:200

# ## Optimization
python run.py --environment logpile --settings-file Test2/settings_optimize_logs.yml --agxOnly --spawner TreeLog:3 \
--controller AddObserver OptimizerBase LogOptimizer # HeightfieldOptimizer:'sam6d_results/detection_pem.json'
# ##

### Visualization after
python run.py --environment logpile --settings-file Test2/settings_view_optimized.yml --spawner TreeLog:3 \
--controller AddObserver DoNothing:120 LogVisualizer DoNothing:60


# # python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:10 \
# # --controller AddObserver LoadLogsFromJSON:'sam6d_results/detection_pem.json' 
# ###
# #####################################

# python run.py --environment logpile --agxOnly --settings-file Test2/eval_gt_init.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:100 PoseEvaluator2
# python run.py --environment logpile --agxOnly --settings-file Test2/eval_gt_opt.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:10 PoseEvaluator2
# python run.py --environment logpile --agxOnly --settings-file Test2/eval_gt_sam6d.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
# python run.py --environment logpile --agxOnly --settings-file Test2/eval_sam6d_init.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
# python run.py --environment logpile --agxOnly --settings-file Test2/eval_sam6d_opt.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator
# python run.py --environment logpile --agxOnly --settings-file Test2/eval_init_opt.yml --spawner TreeLog:3 \
# --controller AddObserver DoNothing:10 LoadLogsFromJSON PoseEvaluator

####################################

deactivate
cd SAM-6D/SAM-6D
