function filelist=nc_generate_file_list(fromdir, folder_prefix, file_prefix, file_suffix, second_folder)
filelist={};

folders=dir(fullfile(fromdir, [folder_prefix, '*']));
if isempty(folder_prefix) == 1
    folders(1:2)=[];
end

for i=1:numel(folders)
    one_dir=fullfile(fromdir, folders(i).name, second_folder);
    file_regexp=[file_prefix, '*', file_suffix];
    files=dir(fullfile(one_dir, file_regexp));
    if isempty(file_prefix) && isempty(file_suffix)
        files(1:2)=[];
    end
    for j=1:length(files)
        fname=fullfile(one_dir, files(j).name);
        if exist(fname, 'file') == 2
            fname=files(j).name;
%             filelist{length(filelist)+1}=fname;
            filelist{length(filelist)+1}=fullfile(one_dir, files(j).name);
%             filelist{length(filelist)+1}=fullfile(files(j).name);
        end
    end
end
        

