% function construct_where_str(protocol, kw, table)
% 
%     % create SQL query based on keyword arguments passed to function
%     where_str = ['protocol="' protocol '" AND exclude=0 AND has_' table '=1']
%     % where_vals = []
% 
%     for key, val in zip(kw.keys(), kw.values()):
%         if val is not None:
%             if table == 'imaging':
%                 if key == 'code':
%                     where_str.append(table + '.name IN(SELECT name FROM mouse WHERE code IN ' + str(val) + ')')
%                 elif key == 'genotype':
%                     where_str.append(table + '.name IN(SELECT name FROM mouse WHERE genotype LIKE "%' + str(val) + '%")')
%                 elif key == 'continuous':
%                     where_str.append(key + '=' + str(val))
%                     % where_vals.append(val)
%                 elif key == 'wavelength':
%                     where_str.append(key + '<=' + str(val))
%             elif table == 'ephys' and key == 'probe1_region':
%                 where_str.append(key + '=' + '"' + str(val) + '"')
% 
%             if key in ['n_trial', 'quality', 'phase']:
%                 where_str.append(key + '>=' + str(val))
%                 % where_vals.append(val)
%             elif key == 'significance':
%                 where_str.append('session.' + key + '=' + str(val))
%                 % where_vals.append(val)
%             elif key == 'curated':
%                 where_str.append(key + '=' + str(val))
%                 % where_vals.append(val)
%             elif key == 'name':
%                 where_str.append(table + '.name IN ' + str(val))
%             elif key == 'exclude_names':
%                 where_str.append(table + '.name NOT IN ' + str(val))
% 
%     where_str = ' AND '.join(where_str)
%     % where_vals = [x for x in kw.values() if type(x) == int]
% 
%     cols = [table + '.figure_path', 'behavior_path', 'file_date_id', table + '.file_date', table +
%             '.processed_data_path', table + '.meta_time', 'stats', 'session.name', 'session.mid', 'sid', 'rid',
%             'session.exp_date', 'session.probe1_AP', 'session.probe1_ML', 'session.probe1_DV', 'session.significance']
% 
%     sql = 'SELECT ' + ', '.join(cols) + ' FROM ' + table + ' LEFT JOIN session ON ' + table + '.behavior_path = ' + \
%           'session.raw_data_path WHERE ' + where_str + ' ORDER BY session.mid ASC, ' + table + '.file_date ASC'
% 
%     return sql
%             end