#!/bin/bash
#
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH --mem=50G

cd scripts

# setup
export DataDirectory="$1"
export OutputDirectory="$1/output"
mkdir -p $OutputDirectory
export Labeling="abeta_5xx"
export ParameterFilePath="/home/users/dadgarki/code/alz-drug-3d/paramfiles/abeta_params.py"
export InputFilePath="$1/input_file.py"
export PostCellDetectionScript="post_cell_detection_script.py"

# run
export Orientation="left"
export BaseDirectory="$OutputDirectory/left/"
python parameter+process_file.py
export LeftFolder=$BaseDirectory
export Orientation="right"
export BaseDirectory="$OutputDirectory/right/"
python parameter+process_file.py
export RightFolder=$BaseDirectory
export OutputFolder="$OutputDirectory/whole/"
python post_process_file.py
