clear all
clc

% set the value to the Pred_folder set in step 2
predfile_dir='';
% coregister individual MRA volume (along with its prediction) to its 
% paired T1 space but with spatial resolution of 0.5 * 0.5 * 0.5
group_t1_dir=nc_GetFullPath('../1_preprocessing/imgs/T1/');
group_mra_dir=nc_GetFullPath('../1_preprocessing/imgs/MRA/');
group_mrapred_dir=predfile_dir;
folder_prefix='Subj';
file_prefix='Subj';
file_suffix='.nii';
subj_folders=nc_generate_folder_list(group_mra_dir, folder_prefix, 0);

% convert to nii format for further processing.
mrapred_filelist=dir(fullfile(group_mrapred_dir,'*.nii.gz'));
for i=1:length(subj_folders)
    [fidr, fname, fext]=fileparts(fullfile(group_mrapred_dir, mrapred_filelist(i).name));
    outfile=fullfile(group_mra_dir, subj_folders{i}, fname);
    save_untouch_nii(load_untouch_nii(fullfile(group_mrapred_dir, mrapred_filelist(i).name)), outfile);
end

coreg_matfile=fullfile(pwd, 'jobmats/Coregister_Estimate_Reslice.mat');
resliced_c1_files=cell(1, length(subj_folders));
for isubj=1:length(subj_folders)
    mra_file=nc_get_specific_file_path(fullfile(group_mra_dir, subj_folders{isubj}), file_prefix, file_suffix);
    mra_label_file=nc_get_specific_file_path(group_mrapred_dir, subj_folders{isubj}, predfile_suffix);
    c1_file=nc_get_specific_file_path(fullfile(group_t1_dir, subj_folders{isubj}), 'c1', predfile_suffix);
    [~, fname, fext]=fileparts(c1_file);
    resliced_c1_files{isubj}=fullfile(group_t1_dir, subj_folders{isubj}, ['resliced_', fname, fext]);
    reslice_nii(c1_file, resliced_c1_files, resliced_voxel);

    interp=4; % 4 for 4-order B-spline; 0 for nearest neighbor;
    nc_spm_coreg_subject(coreg_matfile, resliced_c1_files, {mra_file}, interp);
    interp=0;
    nc_spm_coreg_subject(coreg_matfile, resliced_c1_files, {mra_label_file}, interp, 'ToT1');
end

% convert coregisterred MRA files to .gz format so that they can be fed
% into the deep learning models.
coregpred_dir=fullfile(pwd, 'coregMRAs');
for i=1:length(subj_folders)
    coreg_mrafile=nc_get_specific_file_path(group_mra_dir, 'coreg', file_suffix);
    [fdir, fname, fext]=fileparts(coreg_mrafile);
    outfile=fullfile(coregpred_dir, [fname, fext, '.gz']);
    save_untouch_nii(load_untouch_nii(coreg_mrafile), outfile);
end

