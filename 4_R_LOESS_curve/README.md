# Step 4: Normative models of arterial and cortical volumes
We compute locally estimated scatterplot smoothing (LOESS) to obtain normative models of arterial and cortical volumes of healthy subjects and subjects of diseases.

## System requirements
### Operating System
Windows.

### Hardware requirements
CPU with 4 cores and 8GB RAM are enough. 


## Installation instructions
1. Install python following the instructions in [Step2](../2_CereVessSeg_CereVessPro/README.md):
2. Download this 4_R_LOESS_curve folder to your device.
3. Install required libraries:
   ```
     cd Path_you_save/4_R_LOESS_curve
     pip install -r requirements.txt
	```
   
## How to run it? 
Following previous Steps 1-3, you can extract hierarchical arterial and cortical volumes by yourself using your own dataset. Then you can run our python notebooks to get the normative models of your data by following the below instruction. We also provide CSV files containing arterial and cortical volume features from healthy and pathological subjects in the [data folder](./data/), which you can use to reproduce normative models of our dataset or test.  
1. Check if hierarchical arterial and cortical volumes of each individual are saved in CSV files in the folder 
   PATH_you_save/4_R_LOESS_curve/data;
2. (Optional) Default CSV files of arterial and cortical volumes are listed [here](./data/README.md). If you change these CSV file names, please also remember to change them in every python notebook.
4. Run python notebooks step by step
   - AV_health_curve.ipynb:  
  normative models of arterial volumes for UK healthy subjects and CN healthy subjects regarding to the whole brain, the four typical regions and the brodmann regions;
   - CV_health_curve.ipynb:  
  normative models of cortical volumes for UK healthy subjects and CN healthy subjects regarding to the whole brain, the four typical regions and the brodmann regions;
   - AV_health_vs_unhealth_curve.ipynb:  
  comparison between normative models of arterial volumes for healthy subjects and pathological subjects (AD, large territorial stroke, lacunar stroke); 
   - CV_health_vs_unhealth_curve.ipynb:  
  comparison between normative models of cortical volumes for healthy subjects and pathological subjects (AD, large territorial stroke, lacunar stroke).
