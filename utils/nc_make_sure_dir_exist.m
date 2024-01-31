function nc_make_sure_dir_exist(workdir)
if nc_dir_exist(workdir) == 1
    return;
else
    mkdir(workdir);
end
end

function m=nc_dir_exist(path)
    if exist(path, 'dir') == 7
        m=1;
    else
        m=0;
    end
end
