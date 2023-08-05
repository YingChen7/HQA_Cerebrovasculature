function nc_spm_deform_using_y_file(infile, y_file, voxsize, interpolation)
SPMJOB.matlabbatch{1}.spm.spatial.normalise.write.subj.def = {y_file};
SPMJOB.matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {[infile, ',1']};
SPMJOB.matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72
                                                                    90 90 108];
SPMJOB.matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = voxsize;
SPMJOB.matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = interpolation;
SPMJOB.matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

spm_jobman('run',SPMJOB.matlabbatch);
