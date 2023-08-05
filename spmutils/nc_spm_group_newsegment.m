function nc_spm_group_newsegment(SPMFilePath, newseg_jobmatfile, subj_t1files)
% SPMFilePath='G:\guobin\spm12\';
nsubj=length(subj_t1files);
parfor i=1:nsubj
    [fpath, ~, ~]=fileparts(subj_t1files{i});
    yfile=get_specific_file_path(fpath, 'y_', '');
    if file_exist(yfile)
        continue
    end
    SPMJOB = load(newseg_jobmatfile); 
    for T1ImgSegmentDirectoryNameue=1:6
        SPMJOB.matlabbatch{1,1}.spm.tools.preproc8.tissue(1,T1ImgSegmentDirectoryNameue).tpm{1,1}=[SPMFilePath,filesep,'tpm',filesep,'TPM.nii',',',num2str(T1ImgSegmentDirectoryNameue)]; 
        SPMJOB.matlabbatch{1,1}.spm.tools.preproc8.tissue(1,T1ImgSegmentDirectoryNameue).warped = [0 0]; % Do not need warped results. Warp by DARTEL instead
    end

    SPMJOB.matlabbatch{1,1}.spm.tools.preproc8.warp.affreg = 'mni';
    SPMJOB.matlabbatch{1,1}.spm.tools.preproc8.channel.vols={[subj_t1files{i}]};

    preproc = SPMJOB.matlabbatch{1,1}.spm.tools.preproc8;
    %Set the TPMs
    for T1ImgSegmentDirectoryNameue=1:6
        preproc.tissue(1,T1ImgSegmentDirectoryNameue).tpm{1,1}=[SPMFilePath,filesep,'tpm',filesep,'TPM.nii',',',num2str(T1ImgSegmentDirectoryNameue)];
        preproc.tissue(1,T1ImgSegmentDirectoryNameue).warped = [0 0]; % Do not need warped results. Warp by DARTEL
    end

    %Set the new parameters in SPM12 to default
    preproc.warp.mrf = 1;
    preproc.warp.cleanup = 1;
    preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    preproc.warp.fwhm = 0;
    SPMJOB=[];
    SPMJOB.matlabbatch{1,1}.spm.spatial.preproc = preproc;
    try
        spm_jobman('run',SPMJOB.matlabbatch);
    catch em
        continue;
    end
end
