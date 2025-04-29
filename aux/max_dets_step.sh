#!/bin/bash
# Usage: ./script.sh /path/to/dets_dir
# Example: ./max_dets_step.sh ./input_files/fp_files/fchl_h5/

parent_dir="${1:-.}"

for d in "$parent_dir"/chnl_*; do 
  echo -n "$(basename "$d"): "
  find "$d" -mindepth 2 -type f -name '*_d*_dettable.h5' | 
  awk 'match($0, /step([0-9]+)_d([0-9]+)/, a) {
         if(a[2] > max) { max = a[2]; step = a[1] }
         count++
       } END { print count " steps, max ndets=" max " at step " step }'
done
