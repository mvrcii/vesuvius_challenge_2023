#!/bin/bash

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

# Configuration
outputFolder="/scratch/medfm/vesuv/kaggle1stReimp/data/fragments/slices"
ranges=(
    "00024 00039"
)
overwriteExistingFiles=false

# Create output folder if it doesn't exist
mkdir -p "$outputFolder"

# Function to download a file
download_file() {
    local url=$1
    local outputFile=$2

    if $overwriteExistingFiles || [ ! -f "$outputFile" ]; then
        echo "Downloading file: $outputFile"
        curl -s -u "$credentials" -o "$outputFile" "$url" || echo "Error downloading file: $outputFile"
    else
        echo "File $outputFile already exists, skipping download."
    fi
}

# Start timer
jobStart=$(date +%s)

# Main loop
for range in "${ranges[@]}"; do
    read -r start end <<< "$range"
    for ((i=10#$start; i<=10#$end; i++)); do
        printf -v formattedIndex "%02d" $i
        url="http://dl.ash2txt.org/fragments/Frag1.volpkg/working/54keV_exposed_surface/surface_volume/${formattedIndex}.tif"
        outputFile="${outputFolder}/0$(printf "%04d" $i).tif"
        download_file "$url" "$outputFile" &
    done
done

# Wait for all background processes to finish
wait

# End timer and print job duration
jobEnd=$(date +%s)
echo "Job duration: $((jobEnd - jobStart)) seconds"
