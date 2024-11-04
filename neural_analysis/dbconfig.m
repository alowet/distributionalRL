function db = dbconfig
db.dbfile = '/mnt/clusterhome/dist-rl/session_log.sqlite';
db.configfile = '/mnt/clusterhome/dist-rl/session_log_config.json';
val = jsondecode(fileread(db.configfile));
db.allcols = {};
for i=1:length(val.session_fields)
    db.allcols{end+1} = val.session_fields{i}{1};
end
end