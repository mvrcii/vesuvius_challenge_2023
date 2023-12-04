#!/bin/bash

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <fragmentID> [sliceRange]"
fi

# Assign the first argument to a variable
fragmentID="$1"
string_argument="$2"

IFS=',' read -r -a ranges <<< "$string_argument"

#read -r start end <<< "$string_argument"
# How to use this script:
# 1. Go to bash with "bash"
# 2. Go to root dir
# 3. "./util/download.sh FRAG_ID "00000 00063"
echo $ranges

# Configuration based on option
#outputFolder="/scratch/medfm/vesuv/kaggle1stReimp/data/fragments/fragment${fragmentID}/slices"
outputFolder="data/fragments/fragment${fragmentID}/slices"
baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/${fragmentID}/layers/"

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

# Function to check file sizes
check_file_sizes() {
    local firstFileSize
    local fileSizeMismatch=false
    local mismatchedFiles=()

    for file in "$outputFolder"/*.tif; do
        if [ ! -f "$file" ]; then
            continue
        fi

        if [ -z "$firstFileSize" ]; then
            firstFileSize=$(stat -c%s "$file")
        else
            currentFileSize=$(stat -c%s "$file")
            if [ "$firstFileSize" -ne "$currentFileSize" ]; then
                fileSizeMismatch=true
                mismatchedFiles+=("$file")  # Add only mismatched file paths
            fi
        fi
    done

    if [ "$fileSizeMismatch" = false ]; then
        echo "All files have the same size: $firstFileSize bytes"
    else
        echo "There is a file size mismatch among the downloaded files."
        echo "${mismatchedFiles[@]}"  # Echo only mismatched file paths
    fi
}

# Start timer
jobStart=$(date +%s)

echo "Ranges: ${ranges[*]}"

# Main loop
for range in "${ranges[@]}"; do
    IFS=' ' read -r start end <<< "$range"

    echo "Start: $start, End: $end"

    if ! [[ $start =~ ^[0-9]+$ ]] || ! [[ $end =~ ^[0-9]+$ ]]; then
        echo "Invalid slice range. Please provide numeric start and end values."
        continue # Skip this iteration
    fi

    for ((i=10#$start; i<=10#$end; i++)); do
        printf -v formattedIndex "%05d" $i
        url="${baseUrl}${formattedIndex}.tif"
        outputFile="${outputFolder}/0$(printf "%04d" $i).tif"
        if $overwriteExistingFiles || [ ! -f "$outputFile" ]; then
            echo "About to download: $url to $outputFile"
            # download_file "$url" "$outputFile"  # Temporarily commented out for debugging
        else
            echo "File $outputFile already exists, skipping download."
        fi
    done
done

# Wait for all background processes to finish
wait

# Check file sizes and get mismatched files
IFS=$'\n' mismatchedFiles=($(check_file_sizes))

# Redownload mismatched files
for file in "${mismatchedFiles[@]}"; do
    if [[ $file =~ \.tif$ ]]; then  # Ensure it's a .tif file
        fileName=$(basename "$file")
        index=${fileName:0:5}
        url="${baseUrl}${index}.tif"
        outputFile="$outputFolder/$fileName"
        echo "Redownloading mismatched file: $url"
        download_file "$url" "$outputFile" &
    fi
done

# Wait for all background processes to finish
wait

# Optionally, you can recheck the file sizes after the redownloading
echo "Rechecking file sizes after redownloading:"
check_file_sizes

read -p "Press Enter to Continue"

# End timer and print job duration
jobEnd=$(date +%s)
echo "Job duration: $((jobEnd - jobStart)) seconds"
