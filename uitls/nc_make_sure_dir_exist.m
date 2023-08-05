function nc_make_sure_dir_exist(workdir)
if dir_exist(workdir) == 1
    return;
else
    mkdir(workdir);
end