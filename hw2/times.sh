#!/bin/bash

pattern_num=0
grid_dims=(1024 2048 4096 8192 16384 32786)
num_iterations=1024
block_sizes=(64 128 256 512 1024)
output_on=0

for grid_dim in "${grid_dims[@]}"; do
    printf "grid dimension: $grid_dim\n"
    printf "times: ["
    for block_size in "${block_sizes[@]}"; do
        export TIMEFORMAT="%R"

        real_time=$( TIMEFORMAT="%R"; { \
            time ./gol "$pattern_num" "$grid_dim" "$num_iterations" "$block_size" "$output_on" \
            ; } 2>&1 )
        printf "${real_time},"
    done
    printf "]\n"
done

