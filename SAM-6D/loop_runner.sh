#!/bin/bash

# Base path where your test folders are stored
ORIG_TEST_DIR="/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2_original"
ACTIVE_TEST_DIR="/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/Test2/1"
RESULTS_BACKUP="/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/results"

# List of folders to iterate over (inside Test2_original)
# FOLDERS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" )  # Add as many as you have

for FOLDER in $(seq 1 47); do
    echo "============================"
    echo "Running for folder: $FOLDER"
    echo "============================"

    # 1. Clean up previous test folder
    rm -rf "$ACTIVE_TEST_DIR"

    # 2. Copy next test set
    cp -r "$ORIG_TEST_DIR/$FOLDER" "$ACTIVE_TEST_DIR"

    # 3. Run the actual pipeline
    ./SAM-6D/SAM-6D/run_FULL.sh

    # 4. Store the full processed folder back in Test2_original
    PROCESSED_COPY="$ORIG_TEST_DIR/${FOLDER}_done"
    rm -rf "$PROCESSED_COPY"  # Remove old result if exists
    cp -r "$ACTIVE_TEST_DIR" "$PROCESSED_COPY"


    echo "Finished processing folder: $FOLDER"
done
