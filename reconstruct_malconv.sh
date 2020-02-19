#!/bin/bash

# configurations
FR_RECORD_CYCLES=2000
PROCESSED_TRACES=(
  traces/malconv/processed/cache_accesses.0.csv
  # [Warning, previous results will be over-written]
)

# ----------------------------------------------------------------
#  Run the reconstruction scripts
# ----------------------------------------------------------------
for each_trace in ${PROCESSED_TRACES[@]}; do

  echo "python reconstruct_malconv.py \
    --frcycle $FR_RECORD_CYCLES \
    --tr-file $each_trace"

  python reconstruct_malconv.py \
    --frcycle $FR_RECORD_CYCLES \
    --tr-file $each_trace \
    --verbose

done
