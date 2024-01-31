clear all
clc

tmp=pwd;
cd ..
addpath(genpath(fullfile(pwd, 'utils/')));
addpath(genpath(fullfile(pwd, 'spmutils/')));
cd(tmp);

% set the value to the Pred_folder set in step 2
predfile_dir='C:\Users\ASUS\Desktop\gt\';
predfile_suffix='.nii';
% coregister individual MRA volume (along with its prediction) to its 
% paired T1 space but with spatial resolution of 0.5 * 0.5 * 0.5
resliced_voxel=[0.5 0.5 0.5];
group_t1_dir=nc_GetFullPath('../1_preprocessing/imgs/T1/');
group_mra_dir=nc_GetFullPath('../1_preprocessing/imgs/MRA/');
group_mrapred_dir=predfile_dir;
folder_prefix='Subj';
file_prefix='Subj';
file_suffix='.nii';
subj_folders=nc_generate_folder_list(group_mra_dir, folder_prefix, 0);
for isubj=1:length(subj_folders)
    tmp=dir(fullfile(group_mra_dir, subj_folders{isubj}));
    tmp(1:2)=[];
    for itmp=1:length(tmp)
        [~, ~, file_ext]=fileparts(tmp(itmp).name);
        if strcmp(file_ext,'.gz')
            infile=fullfile(group_mra_dir, subj_folders{isubj}, tmp(itmp).name);
            outfile=fullfile(group_mra_dir, subj_folders{isubj}, tmp(itmp).name(1:end-3));
            save_untouch_nii(load_untouch_nii(infile), outfile);
        end
    end
end
% convert to nii format for further processing.
mrapred_filelist=dir(fullfile(group_mrapred_dir,'*.nii.gz'));
for i=1:length(mrapred_filelist)
    [fidr, fname, fext]=fileparts(fullfile(group_mrapred_dir, mrapred_filelist(i).name));
    outfile=fullfile(group_mra_dir, subj_folders{i}, fname);
    save_untouch_nii(load_untouch_nii(fullfile(group_mrapred_dir, mrapred_filelist(i).name)), outfile);
end

coreg_MRA_dir=nc_GetFullPath('../1_preprocessing/imgs/coregMRA/');
nc_make_sure_dir_exist(coreg_MRA_dir);

coreg_matfile=fullfile(pwd, 'jobmats/Coregister_Estimate_Reslice.mat');
resliced_c1_files=cell(1, length(subj_folders));
for isubj=1:length(subj_folders)
    mra_file=nc_get_specific_file_path(fullfile(group_mra_dir, subj_folders{isubj}), file_prefix, file_suffix);
    mra_label_file=nc_get_specific_file_path(fullfile(group_mrapred_dir, subj_folders{isubj}), file_prefix, predfile_suffix);
    c1_file=nc_get_specific_file_path(fullfile(group_t1_dir, subj_folders{isubj}), 'c1', predfile_suffix);
    [~, fname, fext]=fileparts(c1_file);
    resliced_c1_files{isubj}=fullfile(group_t1_dir, subj_folders{isubj}, ['resliced_', fname, fext]);
    reslice_nii(c1_file, resliced_c1_files{isubj}, resliced_voxel);

    interp=4; % 4 for 4-order B-spline; 0 for nearest neighbor;
    nc_spm_coreg_subject(coreg_matfile, resliced_c1_files{isubj}, {mra_file}, interp);
    interp=0;
    nc_spm_coreg_subject(coreg_matfile, resliced_c1_files{isubj}, {mra_label_file}, interp, 'ToT1');

    % normalize the coregistered MRA files to MNI space to derive a common mask
    % for statistical analysis
    subj_yfile=nc_get_specific_file_path(fullfile(group_t1_dir, subj_folders{isubj}), 'y_', '');
    coreg_MRAfile=nc_get_specific_file_path(fullfile(group_mra_dir, subj_folders{isubj}), 'coreg', '');
    interpolation_bspline=4;
    nc_spm_deform_using_y_file(coreg_MRAfile, subj_yfile, [0.5 0.5 0.5], interpolation_bspline);
    
    nc_make_sure_dir_exist(fullfile(coreg_MRA_dir, subj_folders{isubj}));

    [~, fname, fext]=fileparts(coreg_MRAfile);
    save_untouch_nii(load_untouch_nii(coreg_MRAfile), fullfile(coreg_MRA_dir, subj_folders{isubj}, fname));
end

common_mra_mask_file=nc_GetFullPath('../3_feature_extraction/atlas/common_brainmask.nii');
nc_reslice_probabilitymap_and_extract_common_mask_major_voting(group_mra_dir, ...
    folder_prefix, 'wcoreg', 0, 0.7, common_mra_mask_file, 0.5, ...
    0.5, 0.5, 1);

% convert coregisterred MRA files to .gz format so that they can be fed
% into the deep learning models.
for i=1:length(subj_folders)
    coreg_mrafile=nc_get_specific_file_path(fullfile(coreg_MRA_dir, subj_folders{i}), 'coreg', '.nii');
    if isempty(coreg_mrafile)
        continue;
    end
    [fdir, fname, fext]=fileparts(coreg_mrafile);
    save_untouch_nii(load_untouch_nii(coreg_mrafile), [coreg_mrafile, '.gz']);
end


