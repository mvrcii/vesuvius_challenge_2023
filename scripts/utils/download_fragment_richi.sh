#!/bin/bash

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <fragmentID> [sliceRange]"
    exit 1
fi

# Assign the first argument to a variable
fragmentID="$1"
string_argument="$2"

IFS=',' read -r -a ranges <<< "$string_argument"

# Check if fragmentID ends with "_superseded"
if [[ $fragmentID == *"_superseded" ]]; then
    # Remove "_superseded" suffix for outputFolder
    outputFragmentID=${fragmentID%"_superseded"}
else
    outputFragmentID=$fragmentID
fi


# Configuration based on option
outputFolder="data/fragments/fragment${outputFragmentID}/slices"
#baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/${fragmentID}/layers/"
baseUrl="http://dl2.ash2txt.org/richi-uploads/ThaumatoAnakalyptor/scroll_3/sample_segments2/working_700_small_10/layers/"

# Use the provided or default slice range
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