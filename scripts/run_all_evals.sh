#!/bin/bash

# Define the list files to evaluate
LISTS=(
    "dataset/data/fakeflickr_flux_1_dev_test_list.txt"
    "dataset/data/fakeflickr_sd_3_5_large_test_list.txt"
    "dataset/data/fakeflickr_real_rescaled_test_list.txt"
)

# Run evaluation for each list
for LIST in "${LISTS[@]}"; do
    if [ -f "$LIST" ]; then
        echo "********************************************************"
        echo "Starting evaluation for: $LIST"
        echo "********************************************************"
        
        bash scripts/eval_fakeflickr_full.sh "$LIST"
        
        echo "Finished evaluation for: $LIST"
        echo ""
    else
        echo "Warning: List file $LIST not found skipping."
    fi
done

echo "All evaluations completed."
