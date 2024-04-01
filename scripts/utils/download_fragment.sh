#!/bin/bash

# Description:
# This script is used to download fragments from a specified scroll.
# Usage:
# ./download_fragment.sh <fragmentID> [scrollID] [sliceRange]
# - fragmentID: Mandatory. The ID of the fragment to download.
# - scrollID: Optional. The ID of the scroll, defaults to 1 if not provided.
# - sliceRange: Optional. A comma-separated list of file ranges to download, e.g., "0 64".
# Example:
# ./download_fragment.sh 20231012184423 1 "0 64"

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <fragmentID> [sliceRange]"
    exit 1
fi

# Assign arguments
fragmentID="$1"
scrollID="${2:-1}" # Default scroll ID to 1 if not provided
sliceRange="$3" # Optional slice range

# Translate numeric scroll ID into scroll name
case "$scrollID" in
    1) scrollName="Scroll1" ;;
    2) scrollName="Scroll2" ;;
    3) scrollName="PHerc0332" ;;
    4) scrollName="PHerc1667" ;;
    *) echo "Invalid scroll ID provided"; exit 1 ;;
esac

echo $sliceRange
echo "Processing for Scroll ID: $scrollName with range: $sliceRange"

IFS=',' read -r -a ranges <<< "$sliceRange"

# Handle "_superseded" in fragmentID for outputFolder
if [[ $fragmentID == *"_superseded" ]]; then
    outputFragmentID=${fragmentID%"_superseded"}
else
    outputFragmentID=$fragmentID
fi

# Configuration based on options
outputFolder="fragments/fragment${outputFragmentID}/layers"
baseUrl="http://dl.ash2txt.org/full-scrolls/${scrollName}.volpkg/paths/${fragmentID}/layers/"
overwriteExistingFiles=false

# Create output folder if it doesn't exist
mkdir -p "$outputFolder"

# Function to download a file
download_file() {
    local url=$1
    local outputFile=$2

    if $overwriteExistingFiles || [ ! -f "$outputFile" ]; then
        echo "Downloading: $outputFile Url: $url"
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