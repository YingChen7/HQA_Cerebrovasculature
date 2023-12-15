# Step 4: Normative models of arterial and cortical volumes
We compute locally estimated scatterplot smoothing (LOESS) in R to obtain normative models of arterial and cortical volumes of healthy subjects and subjects of diseases.

## System requirements
##### Operating System
Windows.

##### Hardware requirements
CPU with 4 cores and 8GB RAM are enough. 


## Installation instructions
1. Install R language:
	* download R-4.3.1 (or higher version) installer for windows via this [link](https://cran.rstudio.com/).
	* run the installer with default settings.

2. Install RStudio (to run R scripts):
	* download RStudio installer via this [link](https://posit.co/download/rstudio-desktop/).
	* run the installer with default settings.
	* open RStudio and install packages in the console:  
	   ```
      install.packages(c("ggplot2", "ggthemes","gridExtra", "stringr", "cowplot", "DMwR"))
      ```
3. Download this folder (4_R_LOESS_curve) to local device.
   
## How to run it? 
Following previous Steps 1-3, you can extract hierarchical arterial and cortical volumes by yourself using your own dataset. Then you can run our R scripts to get the normative models of your data by following the below instruction. We also provide CSV files containing arterial and cortical volume features from healthy and AD subjects in the [data folder](./data/), which you can use to reproduce normative models of our dataset or test.  
1. Check if hierarchical arterial and cortical volumes of each individual are saved in CSV files in the folder 
   PATH_you_save/4_R_LOESS_curve/data;
2. In every R script, set the variable dir_ as PATH_you_save/4_R_LOESS_curve;
3. (Optional) Default CSV files of arterial and cortical volumes are listed [here](./data/README.md). If you change these CSV file names, please also remember to change them in every R script.
4. Run R scripts.  
   * plot_health_artery.R:  
  normative models of arterial volumes for healthy subjects regarding to the whole brain and the four typical regions of the brain (ACR, PCR, MCR, CoWR);
   * plot_health_brodmann_artery.R:  
  normative models of arterial volumes for healthy subjects regarding to Brodmann areas;
   * plot_health_cortex.R:  
  normative models of cortical volumes for healthy subjects regarding to the whole brain and the four typical regions of the brain; 
   * plot_health_brodmann_cortex.R:  
  normative models of cortical volumes for healthy subjects regarding to Brodmann areas;
   * plot_health_vs_unhealth_artery.R: 
  comparison between normative models of arterial volumes for healthy subjects and for AD subjects;
   * plot_health_vs_unhealth_cortex.R:  
  comparison between normative models of cortical volumes for healthy subjects and for AD subjects.