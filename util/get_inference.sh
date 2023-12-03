#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <fragment_id> <checkpoint_folder_name> <hostname>"
    exit 1
fi

# Assign the folder name and hostname to variables
FRAGMENT_ID=$1
FOLDER_NAME=$2
HOSTNAME=$3

# Set the server path based on the hostname
if [ "$HOSTNAME" = "vast" ]; then
    SERVER_PATH="~/kaggle1stReimp/inference/results"
elif [ "$HOSTNAME" = "slurm" ]; then
    HOSTNAME="slurmmaster-ls6"
    SERVER_PATH="/scratch/medfm/vesuv/kaggle1stReimp/inference/results"
else
    echo "Wrong hostname"
    exit 1
fi

# Full path on the server
FULL_SERVER_PATH="$SERVER_PATH/fragment$FRAGMENT_ID/$FOLDER_NAME"

# Define the local path where you want to save the folder
LOCAL_PATH="$(pwd)/inference/results/fragment$FRAGMENT_ID"

# Check if the directory exists, create it if not
if [ ! -d "$LOCAL_PATH" ]; then
    mkdir -p "$LOCAL_PATH"
fi

# Using scp to copy the directory. -r flag is used for recursive copy.
scp -r "$HOSTNAME:$FULL_SERVER_PATH" "$LOCAL_PATH"

read -p "Press Enter to Continue"

# Check if scp succeeded
if [ $? -eq 0 ]; then
    echo "Folder copied successfully."
else
    echo "Error in copying folder."
    exit 1
fi
