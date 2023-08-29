%% computation of hierarchical cortical and vascular volumes
% generate whole brain probability mask
% first, transformed the coreg-MRA to MNI space
% second, defined a common spatial region for statistical analysis
group_t1_dir=nc_GetFullPath('../1_preprocessing/imgs/T1/');
subj_folders=nc_generate_folder_list(group_t1_dir, folder_prefix, 0);
first_subj_folder='Subject0001';
brodmann_atlas=nc_GetFullPath('./atlas/Brodmann_atlas.nii');
lobe_atlas=nc_GetFullPath('./atlas/Lobes_atlas.nii');
cow_atlas=nc_GetFullPath('./atlas/CoW_atlas.nii');
brain_mask=nc_GetFullPath('./atlas/common_brainmask.nii');

files_to_warp={};
files_to_warp{1}=brodmann_atlas;
files_to_warp{2}=lobe_atlas;
files_to_warp{3}=cow_atlas;
files_to_warp{4}=brain_mask;
files_to_warp=files_to_warp';
mraseg_dir=nc_GetFullPath('./combined_results/');
parfor isubj=1:length(subj_folders)
    template_file=fullfile(group_t1_dir, [first_subj_folder, '\Template_6.nii']);
    template_mat_file=fullfile(group_t1_dir, [first_subj_folder, '\Template_6_2mni.mat']);
    flowfields={nc_get_specific_file_path(fullfile(group_t1_dir, subj_folders{isubj}), 'u_', '')}';
    warped_files={};
    warped_files{1}=fullfile(group_t1_dir, subj_folders{isubj}, 'warped_brodmann.nii');
    warped_files{2}=fullfile(group_t1_dir, subj_folders{isubj}, 'warped_lobe.nii');
    warped_files{3}=fullfile(group_t1_dir, subj_folders{isubj}, 'warped_cow.nii');
    warped_files{4}=fullfile(group_t1_dir, subj_folders{isubj}, 'warped_group_brainmask.nii');
    warped_files=warped_files';

    y_WarpBackByDARTEL(files_to_warp, ...
        warped_files, ...
        resliced_c1file, ...
        template_file, ...
        template_mat_file, ...
        flowfields{1}, 0);

    % convert to .gz format (a compressed format) just to save storage space.
    save_untouch_nii(load_untouch_nii(warped_files{1}), [warped_files{1}, '.gz']);
    save_untouch_nii(load_untouch_nii(warped_files{2}), [warped_files{2}, '.gz']);
    save_untouch_nii(load_untouch_nii(warped_files{3}), [warped_files{3}, '.gz']);
    save_untouch_nii(load_untouch_nii(warped_files{4}), [warped_files{4}, '.gz']);
end

%% compute hierarchical cortical and vascular volumes
brodmann_img=load_nii(brodmann_atlas).img;
brd_labels=unique(brodmann_img(:));
voxelsize=0.5^3;
brodmann_feature=zeros(2, length(subj_folders), length(brd_labels));
aca_feature=zeros(2, length(subj_folders));
mca_feature=aca_feature;
pca_feature=aca_feature;
cow_feature=aca_feature;
for isubj=1:length(subj_folders)
    mra_seg=fullfile(mraseg_dir, [subj_folders{isubj}, '_segmentation.nii.gz']);
    resliced_c1file=nc_get_specific_file_path(fullfile(raw_t1dir, subj_folders{isubj}),...
        'resliced_', '.nii');

    subj_t1_dir=fullfile(group_t1_dir, subj_folders{isubj});

    warped_brodmann=load_nii(fullfile(subj_t1_dir, 'warped_brodmann.nii')).img;
    warped_lobe=load_nii(fullfile(subj_t1_dir, 'warped_lobe.nii')).img;
    warped_cow=load_nii(fullfile(subj_t1_dir, 'warped_cow.nii')).img;
    warped_brain=load_nii(fullfile(subj_t1_dir, 'warped_group_brainmask.nii')).img;

    warped_brain = cast(warped_brain, 'uint8');
    aca_mask = cast((warped_lobe==1) + (warped_lobe==2), 'uint8');
    mca_mask = cast((warped_lobe==3) + (warped_lobe==4), 'uint8');
    pca_mask = cast((warped_lobe==5) + (warped_lobe==6), 'uint8');
    cow_mask = cast(warped_cow==1, 'uint8');

    cortical_mask=cast(load_nii(resliced_c1file).img>gmthre, 'uint8');
    vascular_mask=cast(load_nii(mra_seg).img, 'uint8');

    % extract lobular features:
    aca_feature(1, isubj)=sum(cortical_mask .* aca_mask .* warped_brain, 'all') * voxelsize;
    mca_feature(1, isubj)=sum(cortical_mask .* mca_mask .* warped_brain, 'all') * voxelsize;
    pca_feature(1, isubj)=sum(cortical_mask .* pca_mask .* warped_brain, 'all') * voxelsize;
    cow_feature(1, isubj)=sum(cortical_mask .* cow_mask .* warped_brain, 'all') * voxelsize;

    aca_feature(2, isubj)=sum(vascular_mask .* aca_mask .* warped_brain, 'all') * voxelsize;
    mca_feature(2, isubj)=sum(vascular_mask .* mca_mask .* warped_brain, 'all') * voxelsize;
    pca_feature(2, isubj)=sum(vascular_mask .* pca_mask .* warped_brain, 'all') * voxelsize;
    cow_feature(2, isubj)=sum(vascular_mask .* cow_mask .* warped_brain, 'all') * voxelsize;

    % extract brodmann features:
    for ibrod=1:length(brd_labels)
        brodmann_feature(1, ibrod, isubj)=sum(cast(warped_brodmann == brd_labels(ibrod), 'uint8')...
            .* cortical_mask .* warped_brain, 'all') * voxelsize;

        brodmann_feature(2, ibrod, isubj)=sum(cast(warped_brodmann == brd_labels(ibrod), 'uint8')...
            .* vascular_mask .* warped_brain, 'all') * voxelsize;
    end
end


