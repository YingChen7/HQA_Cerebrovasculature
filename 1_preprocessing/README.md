# Step 1: Data preprocessing
The first step is very important for the following analysis steps. Please follow our instructions to prepare your data.

## System requirements
### Operating System
Windows.

### Hardware requirements
CPU with 4 cores and 16GB RAM are enough. 


## Installation instructions
1. Install [Matlab](https://www.mathworks.com/help/install/install-products.html), default setting is suitable during Matlab installation

2. Install third party package SPM12 in Matlab
	* download SPM12 zipfile via this [link](https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip).If thers's no download window prompted up, go to this [website](https://www.fil.ion.ucl.ac.uk/spm/software/download/) and then choose to download SPM12 in the blank of "SPM version".
			
	* unzip SPM12 zipfile to any directory you want, wherein scripts are placed.
	* launch Matlab, add the directory of the unzipped file (along with its sub-folders) to the default search path of Matlab.

3. Install third party package DPABI in Matlab:
	* download DPABI zipfile via this [link](https://github.com/Chaogan-Yan/DPABI).
	* unzip DPABI zipfile to any directory you want.
	* launch Matlab, add the directory of the unzipped file (along with its sub-folders) to the default search path of Matlab.

4. Install third party package NIFTI package, version 20140122 or higher in Matlab:
	* download NIFTI zipfile via this [link](https://ww2.mathworks.cn/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
	* unzip NIFTI zipfile to any directory you want.
	* launch Matlab, add the directory of the unzipped file (along with its sub-folders) to the default search path of Matlab.

## How to run it?
1. 	Organize the paired T1-MRA volumes the way we did (**!!Very important**). Suppose you got 1k pairs. It is suggested to name each T1 file as Subject+number-T1.nii, and the paired MRA file as project+number-MRA.nii. Then put these paired files in the [imgs folder](./imgs/) like this: 
    ```
    T1/
    ├── Subject0001/
    │   ├──Subject0001-T1.nii
    ├── Subject0002/
    │   ├──Subject0002-T1.nii
    ├── ...
  
    MRA/
    ├── Subject0001/
    │   ├──Subject0001-MRA.nii
    ├── Subject0002/
    │   ├──Subject0002-MRA.nii
    ├── ...
    ```
2. run main_T1_preprocessing.m:  
	To extract masks for grey matter, white matter, and cerebral spinal fluid, which help create deformation fields between individual and MNI space in Step 3.

3. run main_arrange_MRAfiles.m
	This script will copy the original MRA file from each subject to a target direcroty ./imgs/rawMRAs, which can be input to the segmentation model.
