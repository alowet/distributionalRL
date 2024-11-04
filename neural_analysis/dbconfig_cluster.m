function db = dbconfig
db.dbfile = '/n/home06/alowet/dist-rl/session_log.sqlite';
db.configfile = '/n/home06/alowet/dist-rl/session_log_config.json';
val = jsondecode(fileread(db.configfile));
db.allcols = {};
for i=1:length(val.session_fields)
    db.allcols{end+1} = val.session_fields{i}{1};
end
end