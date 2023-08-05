function nc_copy_file_list(fromdir, folder_prefix, file_prefix, file_suffix, todir, newprefix, insubfolder)
from_folders=dir(fullfile(fromdir, [folder_prefix, '*']));
if isempty(folder_prefix)
    from_folders(1:2)=[];
end
if exist('insubfolder', 'var') == 0
    insubfolder = 1;
end
for i=1:length(from_folders)
    tmp=dir(fullfile(fromdir, from_folders(i).name, [file_prefix, '*', file_suffix]));
    if length(tmp)==0
        warning(['no valid file exists in ', from_folders(i).name, '\n']);
        continue;
    elseif length(tmp)>1
        warning(['too many valid files exist in ', from_folders(i).name, '\n']);
        fprintf('only the first one will be copied.\n')
    end
    from_file=fullfile(tmp(1).folder, tmp(1).name);

    if insubfolder == 1
        to_file_dir=fullfile(todir, from_folders(i).name);
        if dir_exist(to_file_dir) == 0
            mkdir(to_file_dir);
        end
        copyfile(from_file, fullfile(to_file_dir, [newprefix, tmp(1).name]));
    else
        copyfile(from_file, fullfile(todir, [newprefix, tmp(1).name]));
    end
%     copyfile(from_file, fullfile(todir, [newprefix, from_folders(i).name, tmp(1).name]));
end