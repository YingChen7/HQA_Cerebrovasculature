clear all
clc

tmp=pwd;
cd ..
addpath(genpath(fullfile(pwd, 'utils/')));
addpath(genpath(fullfile(pwd, 'spmutils/')));
cd(tmp);

% prefix of the subject folders
folder_prefix='Subj';
% prefix/suffix of corresponding T1 files
file_prefix='Subj';
file_suffix='.nii';
image_dir=fullfile(pwd, 'imgs');
group_mra_dir=fullfile(image_dir, 'MRA/');
raw_mra_toseg=fullfile(pwd, 'rawMRAs');
nc_make_sure_dir_exist(raw_mra_toseg);
raw_mra_files=nc_generate_file_list(group_mra_dir, folder_prefix, ...
    file_prefix, file_suffix, '');

iszipfile=0;
if length(raw_mra_files) == 0
    raw_mra_files=nc_generate_file_list(group_mra_dir, folder_prefix, ...
        file_prefix, '.gz', '');
    iszipfile=1;
end
% change each file to .gz format, which can be fed into deep learning
% models
for i=1:length(raw_mra_files)
    [fdir, fname, fext]=fileparts(raw_mra_files{i});
    if iszipfile == 0
        outfile=fullfile(raw_mra_toseg, [fname, fext, '.gz']);
    else
        outfile=fullfile(raw_mra_toseg, [fname, fext]);
    end
    save_untouch_nii(load_untouch_nii(raw_mra_files{i}), outfile);
end


