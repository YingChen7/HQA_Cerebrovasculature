# CereVessSeg model and CereVessPro contrastive learning for cerebrovascular segmentation
We build a deep learning model named CereVessSeg and a contrastive learning method CereVessPro for cerebrovascular segmentation in MRA volumes
## System requirements
### Operating System
Linux (Ubuntu 18.04, 20.04). 
### Hardware requirements
GPU and CPU are required for model training and inference. For training, RTX 3090 or stronger GPU is recommended. A strong CPU is also required to go along with the GPU (at least 20 cores).


## Installation instructions
1. Install CereVessSeg and CereVessPro in a virtual environment. We use anaconda to create and control virtual environments: 
	* download the Anaconda installer using wget commands:  
         ```
         apt-get install wget
	   wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
         ```  
      or directly download the installer from [Anaconda](https://www.anaconda.com/download#downloads) for linux.
	* verify the hash code integrity of the package:  
		`sha256sum Anaconda3-2023.09-0-Linux-x86_64.sh`
	* run the Anaconda bash shell script:  
		`bash Anaconda3-2023.09-0-Linux-x86_64.sh`
	* after running the bash command, you’ll be welcomed to the Anaconda setup. However, you must review and agree to its license agreement before the installation. Hit Enter to continue. Type 'yes' when a question comes out (to use the default directory to place Anaconda)

2. Install CUDA 11.1 or higher. Here we use CUDA 11.1 for example, which was used in our experiments:
	* (optional) if you installed some old versions of CUDA before, it is better to remove previous installation：
         ```
		!/bin/bash
		sudo apt-get purge *nvidia*
		sudo apt remove --autoremove nvidia-*
		sudo rm /etc/apt/sources.list.d/cuda*
		sudo apt remove --autoremove nvidia-cuda-toolkit
		sudo apt-get autoremove && sudo apt-get autoclean
		sudo rm -rf /usr/local/cuda*
         ```
	* verify gpu cuda availability:  
			`lspci | grep -i nvidia`
	* verify the version of gcc compiler, which is required for CUDA usage:  
			`gcc --version`
	* system update:
         ```
      sudo apt-get update  
		sudo apt-get upgrade 
         ```
	* Install other import packages:  
			`sudo apt-get install g++ freeglut3-dev   build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev`
	* get the PPA repository driver: 
         ```
		sudo add-apt-repository ppa:graphics-drivers/ppa
		sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub  
		echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ |sudo tee /etc/apt/sources.list.d/cuda.list 
		sudo apt-get update
         ```
	* Installing CUDA:
         ```
		sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-1 cuda-drivers`
         ``` 
	* setup path:
         ```
		echo 'export PATH=/usr/local/cuda-11.1/bin:$PATH' >> ~/.bashrc
		echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
		source ~/.bashrc
		sudo ldconfig
         ```
	* install cuDNN:
         ```
		wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.1_20200923/cudnn-11.1-linux-x64-v8.0.4.30.tgz
		tar -xzvf cudnn-11.1-linux-x64-v8.0.4.30.tgz
         ```
	* copy the following files into the cuda toolkit directory:
         ```
		sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.1/include
		sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64/
		sudo chmod a+r /usr/local/cuda-11.1/lib64/libcudnn*
         ```
	* final check:
         ```
		sudo apt install nvidia-cuda-toolkit
		nvidia-smi
		nvcc -V
         ```
3. Install Python 3.9 or higher version by Anaconda.  
    ```
    conda create -n your_env python=3.9
	conda activate your_env
	```
4. Install Pytorch 
    ```  
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
	conda install -c anaconda cudnn
	conda install -c anaconda cudatoolkit
	```
## How to run it? 
   1. Download 2_CereVessSeg_CereVessPro folder to your device
   2. Install CereVessSeg and CereVessPro
   		```
		cd Path_you_save/2_CereVessSeg_CereVessPro/CereVessSeg_CereVessPro_code
    	pip install -e .
	    ``` 
   3. Setting up paths as [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) where you intend to save raw data, preprocessed data and trained models. Please follow the instructions [here]( https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).
   4. Download our trained models and then unzip. Here are two ways to get the trained models: [way1](https://pan.baidu.com/s/1v8X-hEaug5Jsawpra97Tcg?pwd=vhcq), 
   [way2](https://terabox.com/s/1V0DPKO9qrfQ6lkFvh3TJCA).
   5. Put our trained models from ./netstore in the path you set for saving trained models.
   6. Directly run segmentation inference by CereVessSeg:
		```
        cd PATH_you_save/2_CereVessSeg_CereVessPro/CereVessSeg_CereVessPro_code/nnunet/inference
        CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder -o Pred_folder -t 503 -tr nnUNetTrainerMSV2 -m 3d_fullres -f 0 -p nnUNetPlansv2.1
        ```
   7. Directly run inference by finetuned CereVessSeg that was pretrained by CereVessPro:
    	```
   		cd Path_you_save/2_CereVessSeg_CereVessPro/CereVessSeg_CereVessPro_code/nnunet/inference
   		CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder -o Pred_folder -t 503 -tr nnUNetTrainerSSLTune -m 3d_fullres -f 0 -p nnUNetPlansv2.1
   		```

