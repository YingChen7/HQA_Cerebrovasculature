function img=nc_reslice_and_load_image(from_img_file, to_img_file, sp_LR, sp_AP, sp_SI)
    if exist(to_img_file, 'file') ~= 2
        reslice_nii(from_img_file,to_img_file,[sp_LR sp_AP sp_SI], 0);
    end
    nii=load_nii(infile, [], [], [], [], [], 1);
    img=nii.img;
end