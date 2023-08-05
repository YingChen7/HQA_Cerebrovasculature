clear all
clc
% set the value to the Pred_folder set in step 2
coreg_of_predMRA_dir='';
% set the value to the Pred_folder2 set in step 3
pred_of_coregMRA_dir='';
% coregister individual MRA volume (along with its prediction) to its 
% paired T1 space but with spatial resolution of 0.5 * 0.5 * 0.5
folder_prefix='Subj';
group_t1_dir=nc_GetFullPath('../1_preprocessing/imgs/T1/');
subj_folders=nc_generate_folder_list(group_t1_dir, folder_prefix, 0);

filelist1=dir(coreg_of_predMRA_dir, '*.gz');
filelist2=dir(pred_of_coregMRA_dir, '.nii');
outpurdir=fullfile(pwd, 'combined_results');
for isubj=1:length(subj_folders)
    subj_dir=fullfile(group_mrapred_dir, subj_folders{isubj});
    pred_then_coreg=get_specific_file_path(subj_dir, tgt_fileprefix, predictfile_suffix);
    coreg_then_pred=get_specific_file_path(subj_dir, coregfile_prefix, predictfile_suffix);
    nii=load_untouch_nii(fullfile(coreg_of_predMRA_dir, filelist1(isubj).name));
    img1=nii.img;
    img2=load_untouch_nii(fullfile(pred_of_coregMRA_dir, filelist2(isubj).name)).img;
    nii.img=img1+img2;
    save_untouch_nii(nii, fullfile(outpurdir, [subj_folders{isubj}, '_segmentation.nii.gz']));
end


