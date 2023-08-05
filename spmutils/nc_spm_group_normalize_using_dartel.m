function nc_spm_group_normalize_using_dartel(normalize_matfile, TemplateFile, FlowFieldFileList, ...
    gm_files, wm_files, csf_files, fdg_files)
% normalize_matfile : spm job mat for normalization, usually refer to the 
% Dartel_NormaliseToMNI_ManySubjects.mat
% template file : the dartel template file created using 
% flowfield file list : files starting with 'u_' in T1ImgNewSegment of each
% subj
% gm_files : files of each subj starting with 'c1*' in T1ImgNewSegment
% wm_files : files of each subj starting with 'c2*' in T1ImgNewSegment 
% csf_files : files of each subj starting with 'c3*' in T1ImgNewSegment 
% fdg_files : files to normalize along with gm, wm, and csf, but these
% files should be in the same coordinate with the t1files used to generate
% the dartel template. So first of all, coregister these 'fdg_files' to the
% corresponding t1files
SPMJOB = load(normalize_matfile);
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.template=TemplateFile;
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.data.subjs.flowfields=FlowFieldFileList;
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.data.subjs.images{1,1}=gm_files;
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.data.subjs.images{1,2}=wm_files;
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.data.subjs.images{1,3}=csf_files;
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.fwhm=[0 0 0]; % Do not want to perform smooth
SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.preserve = 0;
if nargin == 7
    if isempty(fdg_files) == 1
        error('no values in the last variable');
    end
    SPMJOB.matlabbatch{1,1}.spm.tools.dartel.mni_norm.data.subjs.images{1,4}=fdg_files;
end
spm_jobman('run',SPMJOB.matlabbatch);
end
 