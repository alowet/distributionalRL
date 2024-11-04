function db = dbconfig
db.dbfile = '../data/session_log.sqlite';
db.configfile = '../data/session_log_config.json';
val = jsondecode(fileread(db.configfile));
db.allcols = {};
for i=1:length(val.session_fields)
    db.allcols{end+1} = val.session_fields{i}{1};
end
end