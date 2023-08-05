Cerebrovascular segmentation by CereVessSeg and CereVessPro method

1. Download 2_CereVessSeg_CereVessPro folder to your device
2. cd Path_you_save/2_CereVessSeg_CereVessPro/CereVessSeg_CereVessPro_code
3. pip install -e . 
4. Setting up paths as nnUNet where you intend to save raw data, preprocessed data and trained models. Please follow the instructions in https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md
5. Download trained models and then unzip: 
   way1: https://pan.baidu.com/s/1v8X-hEaug5Jsawpra97Tcg?pwd=vhcq
   way2: https://terabox.com/s/1V0DPKO9qrfQ6lkFvh3TJCA
6. Put our trained models from ./netstore in the path you set for saving trained models
7. Directly run inference by CereVessSeg:
   cd Path_you_save/2_CereVessSeg_CereVessPro/CereVessSeg_CereVessPro_code/nnunet/inference
   CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder -o Pred_folder -t 503 -tr nnUNetTrainerMSV2 -m 3d_fullres -f 0 -p nnUNetPlansv2.1
8. Directly run inference by finetuned CereVessSeg that was pretrained by CereVessPro:
   cd Path_you_save/2_CereVessSeg_CereVessPro/CereVessSeg_CereVessPro_code/nnunet/inference
   CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i MRA_folder -o Pred_folder -t 503 -tr nnUNetTrainerSSLTune -m 3d_fullres -f 0 -p nnUNetPlansv2.1

