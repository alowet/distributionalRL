addpath('../Variance_toolbox');
% protocol = 'Bernoulli';
protocol = 'SameRewDist';

% connect to sqlite database
db = dbconfig;
conn = sqlite(db.dbfile);

% hardcode for now b/c I'm lazy. Ultimatley, I should either directly call
% the Python function that generates this, or rewrite it in Matlab
if strcmp(protocol, 'Bernoulli')
    sql = 'SELECT ephys.figure_path, behavior_path, file_date_id, ephys.file_date, ephys.processed_data_path, ephys.meta_time, stats, session.name, session.mid, sid, rid, session.exp_date, session.probe1_AP, session.probe1_ML, session.probe1_DV, session.significance FROM ephys LEFT JOIN session ON ephys.behavior_path = session.raw_data_path WHERE protocol="Bernoulli" AND exclude=0 AND has_ephys=1 AND phase>=4 AND n_trial>=150 AND quality>=2 AND curated=1 AND session.significance=1 AND probe1_region="striatum" ORDER BY session.mid ASC, ephys.file_date ASC';
elseif strcmp(protocol, 'SameRewDist')
    sql = 'SELECT ephys.figure_path, behavior_path, file_date_id, ephys.file_date, ephys.processed_data_path, ephys.meta_time, stats, session.name, session.mid, sid, rid, session.exp_date, session.probe1_AP, session.probe1_ML, session.probe1_DV, session.significance FROM ephys LEFT JOIN session ON ephys.behavior_path = session.raw_data_path WHERE protocol="SameRewDist" AND exclude=0 AND has_ephys=1 AND phase>=3 AND n_trial>=150 AND quality>=2 AND curated=1 AND session.significance=1 AND probe1_region="striatum" ORDER BY session.mid ASC, ephys.file_date ASC';
elseif strcmp(protocol, 'DistributionalRL_6Odours')
    sql = 'SELECT ephys.figure_path, behavior_path, file_date_id, ephys.file_date, ephys.processed_data_path, ephys.meta_time, stats, session.name, session.mid, sid, rid, session.exp_date, session.probe1_AP, session.probe1_ML, session.probe1_DV, session.significance FROM ephys LEFT JOIN session ON ephys.behavior_path = session.raw_data_path WHERE protocol="DistributionalRL_6Odours" AND exclude=0 AND has_ephys=1 AND phase>=7 AND ephys.name NOT IN ("D1-11", "D1-13") AND n_trial>=150 AND quality>=2 AND curated=1 AND session.significance=1 AND probe1_region="striatum" ORDER BY session.mid ASC, ephys.file_date ASC';
end
    
fetchedData = fetch(conn,sql);
close(conn);

n_sessions = size(fetchedData, 1);
n_tt = 6;

makePlot = 1;
close all;

% params for inc_cells
min_fr = 0.1;  % Hz
max_cv = 1;

savedir = fullfile('fano', sprintf('cv%1.1f', max_cv), protocol);
if ~exist(savedir, 'dir')
    mkdir(savedir);
end

ms_per_trial = 6000;

times = 50:50:5500;
fanoP.boxWidth = 100; fanoP.matchReps = 10; fanoP.alignTime = 1000; fanoP.binSpacing = 0.5;

structData = cell(1, n_tt);
Results = cell(n_sessions, n_tt);
start_cell = 0;

for i_ret=1:size(fetchedData, 1)
% for i_ret=1:2
    
    disp(['Structuring session ' num2str(i_ret)]);
    
    % Filename is the name of the file.
%     filename = '/mnt/hdd1/dist-rl/neural-plots/AL39/20211001/AL39_20211001_spikes.p';
%     name = 'AL39';
%     file_date_id = '20211001';
%     filename = '/mnt/ssd2/dist-rl/neural-plots/AL28/20210416/AL28_20210416_spikes.p';
    
    fig_path = char(fetchedData{i_ret, 1});
    file_date_id = char(fetchedData{i_ret, 3});
    name = char(fetchedData{i_ret, 8});
    filename = fullfile(fig_path, strjoin({name, file_date_id, 'spikes.p'}, '_'));

    if ~exist(fullfile(savedir, name, file_date_id), 'dir')
        mkdir(fullfile(savedir, name, file_date_id));
    end

    fid = py.open(filename,'rb');
    data = py.pickle.load(fid);
    dstruct = struct(data);

    trial_types = uint8(dstruct.trial_types);
    inc_cells = and(double(dstruct.means) > min_fr, double(dstruct.cvs) < max_cv);  % cells to include

    % a disgusting one-liner that works. Sorry mom. Trust me, I checked
    % Basically, what's going on is that I'm converting the array into a native
    % Matlab type, then reshaping it into the shape specified by the Python
    % variable. But this reshaping has to be done row-wise, which essentially
    % means doing the reshaping backwards and then transposing
    shp = dstruct.spks.shape;
    spks = permute(reshape(logical(dstruct.spks.base), shp{3}, shp{2}, shp{1}), [3 2 1]);
    spks = spks(inc_cells, :, :);

    n_neurons = sum(inc_cells);
    n_trace_types = max(trial_types);

    for i_tt=0:n_trace_types-1  % must start from zero b/c trial types are 0-indexed!
        for i_cell=1:n_neurons
%             structData(start_cell + i_cell).spikes = squeeze(spks(i_cell, trial_types == i_tt, :));
%             structData(start_cell + n_neurons*i_tt + i_cell).spikes = ...
%                 squeeze(spks(i_cell, trial_types == i_tt, :));
            structData{i_tt+1}(start_cell + i_cell).spikes = squeeze(spks(i_cell, trial_types == i_tt, :));
        end
        Results{i_ret, i_tt+1} = VarVsMean(structData{i_tt+1}(start_cell + 1: start_cell + n_neurons), times, fanoP);
        % PLOT RESULTS
        if makePlot == 1
            plotP.plotRawF = 1;
            plotP = plotFano(Results{i_ret, i_tt+1}, plotP); % plots everything but the red line for the decay
            saveas(gcf, fullfile(savedir, name, file_date_id, [name '_' file_date_id '_tt_' num2str(i_tt) '.pdf']));
             % TITLE
            plotScatter(Results{i_ret, i_tt+1}, 200);
            saveas(gcf, fullfile(savedir, name, file_date_id, [name '_' file_date_id '_scatter_' num2str(i_tt) '.pdf']));
        end
    end
    start_cell = start_cell + n_neurons;
%     start_cell = start_cell + n_neurons*n_trace_types;
    close all;
end
save(fullfile(savedir, [protocol '_results.mat']), 'Results');

% % COMPUTE FANO FACTOR ETC
% times = 500:50:5500;
% fanoP.boxWidth = 100; fanoP.matchReps = 10; fanoP.alignTime = 1000;
% Results = cell(1, 6);
% 
% for i_tt=1:n_trace_types
%     Results{i_tt} = VarVsMean(structData{i_tt}, times, fanoP);
% 
%     % PLOT RESULTS
%     if makePlot == 1
%         plotP.plotRawF = 1;
%         plotP = plotFano(Results{i_tt}, plotP); % plots everything but the red line for the decay
%         saveas(gcf, fullfile(savedir, ['tt_' num2str(i_tt) '.png']));
%          % TITLE
%         plotScatter(Results{i_tt}, 200);
%         saveas(gcf, fullfile(savedir, ['scatter_' num2str(i_tt) '.png']));
%     end
% end
% save(fullfile(savedir, [protocol '_results.mat']), 'Results');
