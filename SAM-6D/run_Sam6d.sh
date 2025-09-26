#!/bin/bash

source /home/philiph/miniconda3/etc/profile.d/conda.sh
conda activate sam6d

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/bin/virtualenv
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
workon autoscene

deactivate
cd SAM-6D/SAM-6D


########### RUNNING SAM6D ###########
### Render CAD templates
cd Render
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
# python utils/generate_heightfield.py --settings_file settings/default_settings.yml --section TakeSnapshot
# python utils/generate_heightfield.py --depth_path "$DEPTH_PATH" --output "$HF_OUTPUT_PATH" \
# --det_dir "$OUTPUT_DIR/sam6d_results/detection_ism.json" --downsampling 4 --camera_yaml "settings/default_settings.yml"
# ##