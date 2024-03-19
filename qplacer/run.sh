
#!/bin/bash

benchmark=(
    "grid-25" 
    "grid-64" 
    "falcon" 
    "hummingbird" 
    "eagle" 
    "Aspen-11" 
    "Aspen-M" 
    "xtree-53"
)

dir="test"
suffix="wp_wf"
counter=0

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

echo "All selected files processed (Total $counter)"
