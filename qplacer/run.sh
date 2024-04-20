
#!/bin/bash

benchmark=(
    "grid-25" 
    "Aspen-M" 
    "Aspen-11" 
    "eagle" 
    "falcon" 
    "xtree-53"

    # "hummingbird" 
)
dir="test"
# suffix_list=(
#     "wp_wf_02"
#     "wp_wf_03"
#     "wp_wf_04"
# )
suffix_list=(
    "default"
    # "classical"
)
counter=0
for suffix in "${suffix_list[@]}"; do
    for folder in "${benchmark[@]}"; do
        echo -e "\n================================="
        for file in "${dir}/${folder}/${suffix}"/*.json; do
            if [[ -f "$file" ]]; then  # Check if it's a file
                log_file="${dir}/${folder}/${suffix}/$(basename "$file" .json).log"
                echo "Processing $file"
                echo "python qplacer_engine/Placer.py $file > $log_file 2>&1"
                python qplacer_engine/Placer.py "$file" > "$log_file" 2>&1
                echo "Save log file to $log_file"
                echo "---------------------------------"
                ((counter++))
            fi
        done
    done
done
echo "All selected files processed (Total $counter)"
