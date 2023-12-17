#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <checkpoint_folder_name> <hostname>"
    exit 1
fi

# Assign the folder name and hostname to variables
FOLDER_NAME=$1
HOSTNAME=$2

# Set the server path based on the hostname
if [ "$HOSTNAME" = "vast2" ]; then
    SERVER_PATH="~/kaggle1stReimp/checkpoints"
else
    SERVER_PATH="/scratch/medfm/vesuv/kaggle1stReimp/checkpoints"
fi

# Full path on the server
FULL_SERVER_PATH="$SERVER_PATH/$FOLDER_NAME"

# Define the local path where you want to save the folder
LOCAL_PATH="$(pwd)/checkpoints/$FOLDER_NAME"

# Using scp to copy the directory. -r flag is used for recursive copy.
scp -r "$HOSTNAME:$FULL_SERVER_PATH" "$LOCAL_PATH"

# Check if scp succeeded
if [ $? -eq 0 ]; then
    echo "Folder copied successfully."
else
    echo "Error in copying folder."
fi

read -p "Press Enter to continue"
