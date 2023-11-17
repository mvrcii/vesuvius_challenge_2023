#!/bin/bash
set -x

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

hostname=$(hostname)
if [[ "$hostname" == "Marcels-MBP.fritz.box" ]]; then
    baseOutputFolder="/Users/marcel/Documents/Git-Master/Private/kaggle1stReimp/data"
elif [[ "$hostname" == "DESKTOP-LLUPIAQ" ]]; then
    baseOutputFolder ="data"
else
    baseOutputFolder="/scratch/medfm/vesuv/kaggle1stReimp/data"
fi

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <option>"
    echo "Options: frag1, frag2, frag3, frag4, frag5, frag6, frag7"
    exit 1
fi

# Assign the first argument to a variable
option=$1

# Configuration based on option
case $option in
    "frag1")
        outputFolder="data/fragments/fragment20230522181603/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230522181603/layers/"
        ;;
    "frag2")
        outputFolder="data/fragments/fragment20230702185752_superseded/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230702185752_superseded/layers/"
        ;;
    "frag3")
        outputFolder="data/fragments/fragment20230827161847/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230827161847/layers/"
        ;;
    "frag4")
        outputFolder="data/fragments/fragment20230904135535/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230904135535/layers/"
        ;;
    "frag5")
        outputFolder="data/fragments/fragment20230905134255/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230905134255/layers/"
        ;;
    "frag6")
        outputFolder="data/fragments/fragment20230909121925/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20230909121925/layers/"
        ;;
    "frag7")
        outputFolder="data/fragments/fragment20231024093300/slices"
        baseUrl="http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/20231024093300/layers/"
        ;;
    *)
        echo "Invalid option: $option"
        echo "Valid options: frag1, frag2, frag3, frag4, frag5, frag6, frag7"
        exit 1
        ;;
esac

# Other configurations
# default 00023 00039 to get 16 slices
#         00019 00043 to get 24 slices
ranges=(
    #"00023 00039"
#    "00019 00043"
#    "00015 00044"
    "00000 00064"
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
        echo "Test ${outputFolder}"
        outputFile="${outputFolder}/0$(printf "%04d" $i).tif"
        download_file "$url" "$outputFile" &
    done
done

# Wait for all background processes to finish
wait

# End timer and print job duration
jobEnd=$(date +%s)
echo "Job duration: $((jobEnd - jobStart)) seconds"
