clear all
clc

tmp=pwd;
cd ..
addpath(genpath(fullfile(pwd, 'utils/')));
addpath(genpath(fullfile(pwd, 'spmutils/')));
cd(tmp);

% set the value to the Pred_folder set in step 2
coreg_of_predMRA_dir='';
% set the value to the Pred_folder2 set in step 3
pred_of_coregMRA_dir='';
% coregister individual MRA volume (along with its prediction) to its 
% paired T1 space but with spatial resolution of 0.5 * 0.5 * 0.5
folder_prefix='Subj';
group_t1_dir=nc_GetFullPath('../1_preprocessing/imgs/T1/');
outputdir=fullfile(pwd, 'combined_results');
nc_make_sure_dir_exist(outputdir);
subj_folders=nc_generate_folder_list(group_t1_dir, folder_prefix, 0);
for i=1:length(subj_folders)
    coreg_predMRAfile=nc_get_specific_file_path(fullfile(coreg_of_predMRA_dir, subj_folders{i}), 'ToT1', '.nii');
    if isempty(coreg_predMRAfile)
        disp(["No coregisterred file for predicted MRA for subject ", subj_folder{i}, '\n']);
        continue;
    end
    pred_coregMRAfile=nc_get_specific_file_path(fullfile(pred_of_coregMRA_dir, subj_folders{i}), 'coreg', '.gz');
    if isempty(pred_coregMRAfile)
        disp(["No predicted file of coregisterred MRA for subject ", subj_folder{i}, '\n']);
        continue;
    end

    nii=load_untouch_nii(coreg_predMRAfile);
    img1=nii.img;
    img2=load_untouch_nii(pred_coregMRAfile).img;
    nii.img=img1+img2;
    subj_outputdir=fullfile(outputdir, subj_folders{i});
    nc_make_sure_dir_exist(subj_outputdir);
    save_untouch_nii(nii, fullfile(subj_outputdir, [subj_folders{i}, '_segmentation.nii.gz']));
end


