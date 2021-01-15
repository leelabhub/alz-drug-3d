#!/bin/bash
#
#SBATCH -p bigmem
#SBATCH -t 24:00:00
#SBATCH --mem=200G

export BaseDirectory="$1"
cd scripts
python preprocess_stitch_orange.py

