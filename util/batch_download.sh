#!/bin/bash

downloadScript="./util/download.sh"
labelFilesDir="./data/base_label_files/layered"

get_fragment_ids() {
    python3 -c "import sys; sys.path.append('.'); from constants import FRAGMENTS; print(' '.join(FRAGMENTS.values()))"
}

determine_slice_range() {
    local fragmentID=$1
    local files=($(ls $labelFilesDir/$fragmentID/*_inklabels_*.png 2> /dev/null))

    local minSlice=99999
    local maxSlice=0

    for file in "${files[@]}"; do
        if [[ $file =~ _([0-9]+)_([0-9]+)\.png ]]; then
            local start=${BASH_REMATCH[1]}
            local end=${BASH_REMATCH[2]}

            if (( start < minSlice )); then minSlice=$start; fi
            if (( end > maxSlice )); then maxSlice=$end; fi
        fi
    done

    echo "$minSlice $maxSlice"
}

check_downloaded_slices() {
    local fragmentID=$1
    local fragmentDir="./data/fragments/fragment$fragmentID/slices"
    local startSlice=$2
    local endSlice=$3
    local missingSlices=()
    local existingSlices=()
    local sliceSizes=()
    local firstSliceSize

    # Check if the slices directory exists and is a directory
    if [ ! -d "$fragmentDir" ]; then
        >&2 echo "Slices directory does not exist for fragment ID: $fragmentID"
        return 1
    fi

    for ((i=startSlice; i<=endSlice; i++)); do
        printf -v sliceFile "%s/%05d.tif" "$fragmentDir" "$i"
        if [ ! -f "$sliceFile" ]; then
            >&2 echo "Missing slice file: $sliceFile"
            missingSlices+=($i)
        else
            existingSlices+=($i)
            if [ -z "$firstSliceSize" ]; then
                firstSliceSize=$(stat -c%s "$sliceFile")
            fi
            sliceSizes[$i]=$(stat -c%s "$sliceFile")
        fi
    done

    for slice in "${existingSlices[@]}"; do
        if [ "${sliceSizes[$slice]}" -ne "$firstSliceSize" ]; then
            >&2 echo "File size mismatch in slice file: $(printf '%s/%05d.tif' "$fragmentDir" "$slice")"
        fi
    done

    if [ ${#missingSlices[@]} -eq 0 ]; then
        return 0
    else
        >&2 echo "Missing slices for $fragmentID: ${missingSlices[*]}"
        >&2 echo "Existing slices for $fragmentID: ${existingSlices[*]}"
        echo "${missingSlices[*]}"
    fi
}

function get_consecutive_ranges {
    local -a local_missing_slices=("$@")
    local -a ranges
    local start=${local_missing_slices[0]}
    local end=${local_missing_slices[0]}

    for i in "${local_missing_slices[@]:1}"; do
        if (( i == end + 1 )); then
            end=$i
        else
            ranges+=("$(printf "%05d %05d" "$start" "$end")")
            start=$i
            end=$i
        fi
    done
    ranges+=("$(printf "%05d %05d" "$start" "$end")")

    printf "%s\n" "${ranges[@]}"
}

# Main script execution
fragmentIDs=($(get_fragment_ids))
for fragmentID in "${fragmentIDs[@]}"; do
    printf "Fragment ID: $fragmentID -> "

    read startSlice endSlice <<< $(determine_slice_range "$fragmentID")
    if [[ $startSlice -eq 99999 || $endSlice -eq 0 ]]; then
        printf "No label files found -> Skipping download\n"
        continue
    else
        printf "Labels = [$startSlice, $endSlice] -> "
    fi

    missingSlices=$(check_downloaded_slices "$fragmentID" "$startSlice" "$endSlice")
    if [[ -z $missingSlices ]]; then
        >&2 echo "No missing slices found -> Skipping download"
        continue
    else
        "$downloadScript" "$fragmentID" "$(get_consecutive_ranges "$missingSlices")"
    fi
done

echo "Batch download script execution completed."
