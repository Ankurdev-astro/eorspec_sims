#!/bin/bash
# Usage: ./aux/ndets_perchnl.shh /path/to/parent_dir
# Example: ./aux/ndets_perchnl.sh ./input_files/fp_files/fchl_h5/

parent_dir="${1:-.}"

for d in "$parent_dir"/chnl_*; do 
  echo -n "$(basename "$d"): "
  find "$d" -mindepth 2 -type f -name '*_d*_dettable.h5' | \
  awk 'match($0, /_d([0-9]+)_dettable\.h5/, a) {
         total += a[1]; count++
       } END { print count " steps, total ndets = " total }'
done
