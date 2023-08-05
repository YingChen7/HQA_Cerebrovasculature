function nc_spm_group_create_dartel_template(dartel_template_file, rc1FileList, rc2FileList, rc3FileList)
SPMJOB = load(dartel_template_file);
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.warp.images{1,1}=rc1FileList;
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.warp.images{1,2}=rc2FileList;
if nargin > 3
    SPMJOB.matlabbatch{1,1}.spm.tools.dartel.warp.images{1,3}=rc3FileList;
end
spm_jobman('run',SPMJOB.matlabbatch);
end