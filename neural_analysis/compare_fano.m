protocols = {'SameRewDist', 'DistributionalRL_6Odours', 'Bernoulli'};
n_prot = length(protocols);
allResults = cell(1, n_prot);

for i=1:n_prot
    prot = protocols{i};
    tmp = load(fullfile('fano', prot, [prot '_results.mat']));
    allResults{i} = tmp.Results;
end