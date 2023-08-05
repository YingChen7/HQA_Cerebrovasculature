Compiled standalone software and/or source code:

We here provide the source code we wrote using common softwares.

------------------------------------

A small dataset to demo the software/code:

Two subjects with paired T1-MRA volumes are included to demo the computational procedure.
Please refer to the demo section for details.

------------------------------------

System requirements:

software dependencies and operating system:
0. Anaconda, for controlling running environment of deep-learning models.

1. PyTorch 1.9.1 + CUDA 11.1 cuDNN 8 on ubuntu 18.04
(hardware requirment: a NVIDIA Geforce RTX 3090 with 24 GB memory)

2. Matlab2019b (or higher version) and third party Matlab packages, including SPM12, DPABI (v7.0 or higher), and NIFTI (version 20140122 or higher) on Windows 11. 

3. R language 4.2 (or higher version) and RStudio 2023.03.0 Build 386 (or higher version) on Windows 11.

------------------------------------
Installation guide:
0. Anaconda
	* install wget first:
		apt-get install wget
	* download the Anaconda installer:
		wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
	* verify the hash code integrity of the package:
		sha256sum Anaconda3-2022.05-Linux-x86_64.sh
	* run the Anaconda bash shell script:
		bash Anaconda3-2022.05-Linux-x86_64.sh
	* after running the bash command, you’ll be welcomed to the Anaconda setup. However, you must review and agree to its license agreement before the installation. Hit Enter to continue.	Type "yes" when a question comes out (to use the default directory to place Anaconda). 

1. PyTorch and CUDA toolkit:
	Installation of CUDA (prior to PyTorch):
		* remove previous installation first：
			!/bin/bash
			sudo apt-get purge *nvidia*
			sudo apt remove --autoremove nvidia-*
			sudo rm /etc/apt/sources.list.d/cuda*
			sudo apt remove --autoremove nvidia-cuda-toolkit
			sudo apt-get autoremove && sudo apt-get autoclean
			sudo rm -rf /usr/local/cuda*
		* verify gpu cuda availability:
			lspci | grep -i nvidia
		* verify the version of gcc compiler, which is required for CUDA usage:
			gcc --version
		* system update:
			sudo apt-get update
			sudo apt-get upgrade 
		* Install other import packages:
			sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
		* get the PPA repository driver:
			sudo add-apt-repository ppa:graphics-drivers/ppa
			sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
			echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
			sudo apt-get update
		* Installing CUDA-11.1:
			sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-1 cuda-drivers
		* setup path:
			echo 'export PATH=/usr/local/cuda-11.1/bin:$PATH' >> ~/.bashrc
			echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
			source ~/.bashrc
			sudo ldconfig
		* install cuDNN v8.0.4:
			CUDNN_TAR_FILE="cudnn-11.1-linux-x64-v8.0.4.30.tgz"
			wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.1_20200923/cudnn-11.1-linux-x64-v8.0.4.30.tgz
			tar -xzvf ${CUDNN_TAR_FILE}
		* copy the following files into the cuda toolkit directory:
			sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.1/include
			sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64/
			sudo chmod a+r /usr/local/cuda-11.1/lib64/libcudnn*
		* final check:
			sudo apt install nvidia-cuda-toolkit
			nvidia-smi
			nvcc -V

	Installation of PyTorch:
		pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
		conda install -c anaconda cudnn
		conda install -c anaconda cudatoolkit

2. Matlab and third party packages:
	Installation of Matlab:
		default setting is suitable for Matlab installation

	SPM12 package:
		* download SPM12 zipfile via this link:
			https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip
			if thers's no download window prompted up, change to following link and fill some blanks before it starts to download. But remember to choose SPM12 in the blank of "SPM version".
			https://www.fil.ion.ucl.ac.uk/spm/software/download/
		* unzip the file to desired directory, e.g. C:\Users\guobin\Desktop\spm12\, wherein scripts are placed.
		* launch Matlab, add this directory (along with its sub-folders) to the default search path of Matlab.
		* done!

	DPABI package:
		* download DPABI zipfile via this link:
			https://github.com/Chaogan-Yan/DPABI
		* unzip this file to desired directory, e.g. C:\Users\guobin\Desktop\DPABIv7.0\, wherein scripts are placed.
		* launch Matlab, add this directory (along with its sub-folders) to the default search path of Matlab.
		* done!

	NIFTI package, version 20140122 or higher:
		* download NIFTI zipfile via this link:
			https://ww2.mathworks.cn/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
		* unzip this file to desired directory, e.g. C:\Users\guobin\Desktop\NIFTI20140122\, wherein scripts are placed.
		* launch Matlab, add this directory (along with its sub-folders) to the default search path of Matlab.
		* done!

3. R language:
		* download the installation program via this link:
			https://cran.rstudio.com/
			select "download R for windows" -> "install R for the first time" -> "Download R-4.3.1 for Windows"
		* run the installation program with default settings.
		* done!

4. RStudio (to run R scripts):
		* download RStudio installation program via this link:
			https://posit.co/download/rstudio-desktop/
		* run the installation program with default settings.
		* done!
5. folders including utils and spmutils should be added to the default search path of Matlab.

------------------------------------

Demo

A subset consisting of 2 paired T1-MRA volumes was used in following running example to show the procedure of extracting features (i.e. cortical and vascular volumes).

Please run following scripts sequentially:
1. Preprocessing
	* run main_T1_preprocessing.m:
		To extract masks for grey matter, white matter, and cerebral spinal fluid, which help create deformation fields between individual and MNI space.

	* run main_arrange_MRAfiles.m
		This script will copy the original MRA file from each subject to target direcroty D1, which can be input to the segmentation model.

2. Segmentation
	* run command line with two parameters: MRA_folder and Pred_folder:
		CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder -o Pred_folder -t 503 -tr nnUNetTrainerSSLTune -m 3d_fullres -f 0 -p nnUNetPlansv2.1 
		MRA_folder should be the absolute path of the folder rawMRAs located in 1_preprocessing.
		Pred_folder can be defined according to the user's willingness.

	Segmentation will be performed for MRA volumes from MRA_folder, results of which will be saved to Pred_folder. (Here we denote the segmentation results as A.)

3. Feature Extraction:
	* run main_coreg_MRA_to_T1.m:
		Please remember to set the predfile_dir in this file to Pred_folder used above.
		This script coregisters raw MRA (along with its prediction) to individual T1 space.
		Also, copy the coregisterred MRA volumes to target directory coregMRAs, located in 3_feature_extraction.

	* run command line with two parameters: MRA_folder2 and Pred_folder2:
			CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder2 -o Pred_folder2 -t 503 -tr nnUNetTrainerSSLTune -m 3d_fullres -f 0 -p nnUNetPlansv2.1 
			MRA_folder2 should be the absolute path of coregMRAs, located in 3_feature_extraction.
			Pred_folder2 can be defined according to the user's willingness.

	 Segmentation will be performed for MRA volumes from MRA_folder2, results of which will be saved to Pred_folder2. (Here we denote the segmentation results as A.)

	* run main_combine_vascular_segmentations.m：
		Please remember to set the predfile_dir in this file to Pred_folder2 used above.
		This script combines segmentation results from A and B.

	* run main_extract_features.m
		To extract hierarchical cortical and vascular volumes from predefined atlas.
		Features extracted in this script can be saved in a .cvs file for further statistical analysis.

4. Normative modelling:
	* step 1 to 3 help extract hierarchical cortical and vascular features.
	* for illustration of normative modelling, we provide csv files containing cortical and vasuclar features from healthy and AD subjects. 

Here we include files recording hierarchical cortical and vascular volumes. To be specific:
1. healthy_vascular_whole.csv: whole-brain vascular volumes were recorded for 963 healthy subjects
2. healthy_vascular_brodmann.csv: vascular volumes for brodmann areas were recorded for 963 healthy subjects
3. healthy_vascular_lobes.csv: lobular vascular volumes were recorded for 963 healthy subjects
4. healthy_vascular_cow.csv: vascular volumes from Circle of Willis were recorded for 963 healthy subjects
5. healthy_cortical_whole.csv: whole-brain cortical volumes were recorded for 963 healthy subjects
6. healthy_cortical_brodmann.csv: cortical volumes for brodmann areas were recorded for 963 healthy subjects
7. healthy_cortical_lobes.csv: lobular cortical volumes were recorded for 963 healthy subjects
8. healthy_cortical_cow.csv: cortical volumes from Circle of Willis were recorded for 963 healthy subjects
9. AD_vascular_whole.csv: whole-brain vascular volumes were recorded for 147 AD subjects 
10.AD_vascular_lobes.csv: lobular vascular volumes were recorded for 147 AD subjects 
11.AD_vascular_cow.csv: vascular volumes from Circle of Willis were recorded for 147 AD subjects 
12.AD_cortical_whole.csv: whole-brain cortical volumes were recorded for 147 AD subjects 
13.AD_cortical_lobes.csv: lobular cortical volumes were recorded for 147 AD subjects 
14.AD_cortical_cow.csv: cortical volumes from Circle of Willis were recorded for 147 AD subjects 

Please refer to the 4_R_LOESS_curve folder for details. 

------------------------------------

Instructions for use:
	Organize the paired T1-MRA volumes the way we did. Suppose you got 1k pairs. It is suggested to name each T1 file as Subject+number-T1.nii, and project+number-MRA.nii for the paired MRA file. e.g.:
		T1
		....Subject0001
			....Subject0001-T1.nii
		....Subject0002
			....Subject0002-T1.nii
		...
		....Subject1000
			....Subject1000-T1.nii

		MRA
		....Subject0001
			....Subject0001-MRA.nii
		....Subject0002
			....Subject0002-MRA.nii
		...
		....Subject1000
			....Subject1000-MRA.nii

	In this case, three variables including (folder_prefix, file_prefix, first_subj_folder) should be changed to ('Subj', 'Subj', 'Subject0001'), respectively. And you are good to go.


github: 
	https://github.com/YingChen7/HQA_Cerebrovasculature
