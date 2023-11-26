function folder_list=nc_generate_folder_list(fromdir, folder_prefix, IsAbsoluteDir)
tmps=dir(fullfile(fromdir, [folder_prefix, '*']));
if isempty(folder_prefix)
    tmps(1:2)=[];
end
folder_list={};
for i=1:length(tmps)
    pdir=fullfile(fromdir, tmps(i).name);
    if dir_exist(pdir) == 1
        if IsAbsoluteDir == 1
            folder_list{length(folder_list)+1}=pdir;
        else
            folder_list{length(folder_list)+1}=tmps(i).name;
        end
    end
end



