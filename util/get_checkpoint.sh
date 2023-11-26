#!/bin/bash

# Check if a folder name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <checkpoint_folder_name>"
    exit 1
fi

# Assign the folder name to a variable
FOLDER_NAME=$1

SERVER_PATH="/scratch/medfm/vesuv/kaggle1stReimp/checkpoints"

# Full path on the server
FULL_SERVER_PATH="$SERVER_PATH/$FOLDER_NAME"

# Define the local path where you want to save the folder
LOCAL_PATH="$(pwd)/checkpoints/$FOLDER_NAME"


# Using scp to copy the directory. -r flag is used for recursive copy.
scp -r "slurmmaster-ls6:$FULL_SERVER_PATH" "$LOCAL_PATH"

# Check if scp succeeded
if [ $? -eq 0 ]; then
    echo "Folder copied successfully."
else
    echo "Error in copying folder."
fi

