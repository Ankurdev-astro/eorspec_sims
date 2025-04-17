#!/bin/bash

# Directory containing the files
dir="gen_schedules"

# Iterate over all files in the directory
for file in "$dir"/*; do
    echo "Head of $file:"
    head "$file"
    echo "-----------------------------------------"
    echo "-----------------------------------------"

done

