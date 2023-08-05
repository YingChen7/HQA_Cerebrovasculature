function nc_spm_coreg_subject(coreg_matfile, t1_file, other_files, interp, prefix)
% interp: 0 for nearest neighbor; 4 for 4-order B-spline
SPMJOB = load(coreg_matfile);
SPMJOB.matlabbatch{1, 1}.spm.spatial.coreg.estwrite.ref={t1_file};
SPMJOB.matlabbatch{1, 1}.spm.spatial.coreg.estwrite.source=other_files(1,:);
if exist('prefix', 'var') == 0
    prefix='coreg';
end
SPMJOB.matlabbatch{1, 1}.spm.spatial.coreg.estwrite.roptions.prefix=prefix;
SPMJOB.matlabbatch{1, 1}.spm.spatial.coreg.estwrite.other=other_files(1,:);
SPMJOB.matlabbatch{1, 1}.spm.spatial.coreg.estwrite.roptions.interp=interp;
spm_jobman('run',SPMJOB.matlabbatch);