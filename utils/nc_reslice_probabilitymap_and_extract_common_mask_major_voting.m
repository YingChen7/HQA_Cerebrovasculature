function result=nc_reslice_probabilitymap_and_extract_common_mask_major_voting(workdir, subj_folder_prefix, ...
    wm_file_prefix, subj_mask_threshold, voting_threshold, out_file, sp_LR, sp_AP, sp_SI, IsRegenrate)
result=[];
wm_nii=[];
subj_folders=dir(fullfile(workdir, [subj_folder_prefix, '*']));
if isempty(subj_folder_prefix)
    subj_folders(1:2)=[];
    fprintf([num2str(length(subj_folders)), 'subjects in total.']);
end
if length(subj_folders)==0
    error('no valid subj folders');
end
if exist(out_file, 'file') == 2 && IsRegenrate ~= 1
    nii=load_nii(out_file);
    result=nii.img;
    return;
end
for i=1:length(subj_folders)
    fprintf(['processing subj : ', num2str(i), '\n']);
    tmp=dir(fullfile(workdir, subj_folders(i).name, [wm_file_prefix, '*']));
    if isempty(wm_file_prefix)
        tmp(1:2)=[];
    end
    if length(tmp) < 1
        error('no valid White Matter file');
    elseif length(tmp) > 1
        error (['too many candidate white matter files in ', subj_folders(i).name]);
    end
    
    reslice_output_file=fullfile(workdir, subj_folders(i).name, ['tmp_reslice_', tmp(1).name]);
    wm_img=nc_reslice_and_load_image(fullfile(workdir, subj_folders(i).name, tmp(1).name), ...
        reslice_output_file, sp_LR, sp_AP, sp_SI);
    if isempty(wm_nii)
        wm_nii=load_nii(reslice_output_file);
    end
    
    wm_img(isnan(wm_img))=0;
    if isempty(result)
        result=zeros([size(wm_img), length(subj_folders)]);
    end
    result(:,:,:,i)=double(wm_img>subj_mask_threshold);

    delete(reslice_output_file);
end
result=squeeze(mean(result, 4));
result=(result>=voting_threshold);
wm_nii.img=result;
wm_nii.hdr.dime.datatype=16;
wm_nii.hdr.dime.bitpix=32;
save_nii(wm_nii, out_file);
end

