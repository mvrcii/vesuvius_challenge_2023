#!/bin/bash

# Basic authentication
user="registeredusers"
password="only"
credentials="$user:$password"

hostname=$(hostname)
if [[ "$hostname" == "Marcels-MBP.fritz.box" ]]; then
    baseOutputFolder="/Users/marcel/Documents/Git-Master/Private/kaggle1stReimp/data"
elif [[ "$hostname" == "DESKTOP-LLUPIAQ" ]]; then
baseOutputFolder="C:\Users\Marce\Git-Master\Privat\kaggle1stReimp\data"
else
    baseOutputFolder="/scratch/medfm/vesuv/kaggle1stReimp/data"
fi

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <option>"
    echo "Options: fragment1, fragment2, fragment3, fragment4, scroll1"
    exit 1
fi

# Assign the first argument to a variable
option=$1

# Configuration based on option
case $option in
    "fragment1")
        outputFolder="${baseOutputFolder}/fragments/fragment1/slices"
        baseUrl="http://dl.ash2txt.org/fragments/Frag1.volpkg/working/54keV_exposed_surface/surface_volume/"
        ;;
    "fragment2")
        outputFolder="${baseOutputFolder}/fragments/fragment2/slices"
        baseUrl="http://dl.ash2txt.org/fragments/Frag2.volpkg/working/54keV_exposed_surface/surface_volume/"
        ;;
    "fragment3")
        outputFolder="${baseOutputFolder}/fragments/fragment3/slices"
        baseUrl="http://dl.ash2txt.org/fragments/Frag3.volpkg/working/54keV_exposed_surface/surface_volume/"
        ;;
    "fragment4")
        outputFolder="${baseOutputFolder}/fragments/fragment4/slices"
        baseUrl="http://dl.ash2txt.org/fragments/Frag4.volpkg/working/54keV_exposed_surface/PHercParis1Fr39_54keV_surface_volume/"
        ;;
    "scroll1")
        outputFolder="${baseOutputFolder}/segments/scroll1recto/slices"
        baseUrl="http://dl.ash2txt.org/stephen-parsons-uploads/recto/Scroll1_part_1_wrap_recto_surface_volume/"
        ;;
    *)
        echo "Invalid option: $option"
        echo "Valid options: fragment1, fragment2, fragment3, fragment4, scroll1"
        exit 1
        ;;
esac

# Other configurations
# default 00023 00039 to get 16 slices
#         00019 00043 to get 24 slices
ranges=(
    #"00023 00039"
#    "00019 00043"
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