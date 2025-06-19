#!/bin/bash
# set the paths
export CAD_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/Perspective/cyl2.ply   # path to a given cad model(mm)
export RGB_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/snapshot_rgb.png         # path to a given RGB image
export DEPTH_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/snapshot_depth.png       # path to a given depth map(mm)
export CAMERA_PATH=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/camera_logs.json    # path to given camera intrinsics
export OUTPUT_DIR=/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction         # path to a pre-defined file for saving results
export BLENDER_PATH=/home/philiph/Blender/blender-3.3.1-linux-x64

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

python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:3 \
--controller DoNothing:30 DropLogs DoNothing:inf 

# python run.py --environment logpile --settings-file settings/default_settings.yml --spawner TreeLog:4 \
# --controller DropAndEmbedLogs LogStateRecorder TakeSnapshot
##########################################



deactivate
cd SAM-6D/SAM-6D


