#!/bin/bash

for folder in inputs/*/; 
do
    eq_id_path="${folder%/}"
    echo $eq_id_path
    echo "2" > input.txt #mode option: 1-two points, 2-two files
    echo "\"$eq_id_path/event_input.evt\"" >> input.txt #source file
    echo "\"$eq_id_path/station_input.sta\"" >> input.txt #receiver file
    echo "1" >> input.txt #output raypath-1, otherwise 0
    echo "1" >> input.txt #output type 1-ascii, 2-binary
    #Fortran tracer
    ./tracer < input.txt

    output_file="$eq_id_path/output.table"
    cp "tt.table" $output_file
    echo "=========================="
done


