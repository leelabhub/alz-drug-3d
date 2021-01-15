# Region-specific activity of anti-Aβ compounds uncovered by 3D microscopy

Code for reproducing all analysis and figures from Kirschenbaum et al. 2020 ("Quantitative 3D microscopy reveals a genetic network predicting the local activity of anti-Aβ compounds"). Includes scripts for performing 2-channel stitching, registration, segmentation, statistics, and colocalization analysis of various anti-Aβ compounds.

## Requirements:
  - Python v2.7, with packages:
    - numpy, scipy, matplotlib, pandas
    - statsmodels
    - scikit-image
    - ClearMap (https://github.com/ChristophKirst/ClearMap)
  - Additional:
    - elastix (tested with version 4.8)
    - Ilastik
    - TeraStitcher (tested with version 1.10.12)
    - ImageJ
  - Hardware: access to at least 50 GB of RAM

## Instructions

Update [Settings.py](https://github.com/dadgarki/alz-drug-3d/blob/master/scripts/Settings.py) with the correct paths to the elastix, Ilastik, TeraStitcher, and ImageJ executables.

To process a single sample (containing both left and right hemispheres), execute the bash script process.sh and pass in the path to the sample directory

    $ ./process.sh /path/to/sample_01

The directory must contain a configuration file named input_file.py (see [input_file_template.py](https://github.com/dadgarki/alz-drug-3d/blob/master/paramfiles/input_file_template.py)). This file specifies the relative paths to each hemisphere's primary signal and autofluorescence channel, image resolution (um), and the correct axis permutation for aligning the data to the Allen Atlas.
