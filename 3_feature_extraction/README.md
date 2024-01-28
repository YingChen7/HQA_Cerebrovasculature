# Step 3: Hierarchically arterial and cortical volume feature extration
We extract arterial and cortical features of every subject in terms of 3 hierarchical levels: 1) whole brain, 2) four common vascular regions including anterior cortical region (ACR),
posterior cortical region (PCR), middle cortical region (MCR), and circle of willis region (CoWR),
and 3) conventional 82 Brodmann areas.

## System requirements
### Operating System
Windows 10.

### Hardware requirements
CPU with 4 cores and 16GB RAM are enough. 

## Installation instructions
1. Install Matlab and third party packages SPM12, DPABI and NIFTI following the instructions in [Step 1](../1_preprocessing/README.md#installation-instructions), if you haven't installed them.
2. Download this 3_feature_extraction folder under the same directory as 1_preprocessing folder to your device.

## How to run it?
1. Copy the predicted MRA segmentations from the `Pred_folder` in [Step2](../2_CereVessSeg_CereVessPro/README.md#how-to-run-it) to current Windows device.
2. Coregister raw MRA volumes (along with their prediction) to individual T1 space.
   * set the variable `predfile_dir` to the directory for saving predicted segmentations in main_coreg_MRA_to_T1.m.   
   * run main_coreg_MRA_to_T1.m, the resulted every paired coregisterred MRA volume and segmentation will be saved in the corresponding [`./1_preprocessing/MRA/subjectXXXX` folder](../1_preprocessing/imgs/MRA/). And all resulted coregisterred MRA volumes will be further copied to a subfolder `./coregMRAs`.
   * copy all coregisterred MRA segmentations to a folder.

3. Run segmentation inference of all coregisterred MRA volumes following [Step 2](../2_CereVessSeg_CereVessPro/README.md). 
   * copy the above resulted coregisterred MRA volumes to a computer or server with linux system.
   * run command line:
		```
	    CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder -o Pred_folder2 -t 503 -tr nnUNetTrainerSSLTune -m 3d_fullres -f 0 -p nnUNetPlansv2.1 
		```
	 Here `MRA_folder` should be the absolute path of coregisterred MRA volumes. `Pred_folder` can be defined according to the user's willingness. Segmentation inference of all coregisterred MRA volumes will be saved in the `Pred_folder`.
	* copy segmentation inference back.

4. Combine coregisterred MRA segmentation and segmentation for coregisterred MRA volume:
   * set the variable `pred_of_coregMRA_dir` to the folder containing segmentation results of all coregisterred MRA volumes in 3.
   * set the variable `coreg_of_predMRA_dir` to the folder containing all coregisterd MRA segmentation in 2.  
   * run main_combine_vascular_segmentations.m to combine the two segmentation results for all subjects.
	

5. Extract hierarchical cortical and vascular volumes from predefined atlas.
    * run main_extract_features.m. Extracted features will be saved in a .cvs file for further statistical analysis.
