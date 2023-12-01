#!/bin/bash
set -x

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <fragmentID>"
    exit 1
fi

# Assign the first argument to a variable
fragmentID=$1

# Configuration based on option
outputFolder="/scratch/medfm/vesuv/kaggle1stReimp/data/fragments/fragment${fragmentID}/slices"
baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/${fragmentID}/layers/"

# Other configurations
# default 00023 00039 to get 16 slices
#         00019 00043 to get 24 slices
ranges=(
    #"00023 00039"
#    "00019 00043"
#    "00015 00044"
    "00028 00043"
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
        curl -s -u "$credentials" -o "$outputFile" "$url" || echo "Error downloading file: $outputFile $url"
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
        url="${baseUrl}${formattedIndex}.tif"
        outputFile="${outputFolder}/0$(printf "%04d" $i).tif"
        download_file "$url" "$outputFile" &
    done
done

# Wait for all background processes to finish
wait

# End timer and print job duration
jobEnd=$(date +%s)
echo "Job duration: $((jobEnd - jobStart)) seconds"
