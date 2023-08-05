function fpath=nc_get_specific_file_path(fromdir, file_prefix, file_suffix)
tmp=dir(fullfile(fromdir, [file_prefix, '*', file_suffix]));
if length(tmp)>1
    
%     error('too many valid files exist.');
elseif length(tmp) == 0
    fpath='';
    return;
end
if isempty(tmp)
    fpath=[];
    return;
end
fpath=fullfile(fromdir, tmp(1).name);

