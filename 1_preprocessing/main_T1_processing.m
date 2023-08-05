clear all
clc
% installed directory of spm: e.g. G:\guobin\spm12\
SPMdir='C:\Program Files\MATLAB\spm12\'; 
image_dir=fullfile(pwd, 'imgs');
group_t1_dir=fullfile(image_dir, '/T1/');
newseg_jobmatfile=fullfile(pwd, 'jobmats/NewSegment.mat'); 
create_template_jobmatfile=fullfile(pwd, 'jobmats/Dartel_CreateTemplate.mat'); 
normalize_matfile=fullfile(pwd, 'jobmats/Dartel_NormaliseToMNI_ManySubjects.mat');
% prefix of the subject folders
folder_prefix='Subj';
% prefix/suffix of corresponding T1 files
file_prefix='Subj';
file_suffix='.nii';
% threshold to binarize individual gray matter mask
gmthre=0.5;
% sort the subject folders according to their names
% this is the folder ranked in the first
% wherein some critical files are saved
first_subj_folder='Subject0001';

%% resliced all T1 files to resolution space of 0.5 * 0.5 * 0.5 mm3
resliced_voxel=[0.5 0.5 0.5];
subj_folders=nc_generate_folder_list(group_t1_dir, folder_prefix, 0);
subj_t1files=cell(1, length(subj_folders));
for isubj=1:length(subj_folders)
    subj_t1files{isubj}=nc_get_specific_file_path(fullfile(group_t1_dir, ...
        subj_folders{isubj}), file_prefix, file_suffix);
end

%% segmentation of gray matter (GM), white matter (WM), and cerebral spinal fluid (CSF)
nc_spm_group_newsegment(SPMdir, newseg_jobmatfile, subj_t1files);

%% create flow fields using segmentation results
rc1FileList=nc_generate_file_list(group_t1_dir, file_prefix, 'rc1', file_suffix, '');
rc2FileList=nc_generate_file_list(group_t1_dir, file_prefix, 'rc2', file_suffix, '');
rc3FileList=nc_generate_file_list(group_t1_dir, file_prefix, 'rc3', file_suffix, '');
nc_spm_group_create_dartel_template(create_template_jobmatfile, rc1FileList', rc2FileList', rc3FileList');

%% apply flow fields to transform probability maps of GM, WM and CSF to MNI space
template_file=fullfile(group_t1_dir, [first_subj_folder, '\Template_6.nii']);
flowfields=nc_generate_file_list(group_t1_dir, file_prefix, 'u_', '', '')';
wm_files=nc_generate_file_list(group_t1_dir, file_prefix, 'c2', '', '')';
gm_files=nc_generate_file_list(group_t1_dir, file_prefix, 'c1', '', '')';
csf_files=nc_generate_file_list(group_t1_dir, file_prefix, 'c3', '', '')';
nc_spm_group_normalize_using_dartel(normalize_matfile, {template_file}, flowfields, ...
    gm_files, wm_files, csf_files)







