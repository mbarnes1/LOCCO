clear, clc, close all

git = getGitInfo();
git = git.hash(1:6);
notes = '';

%% Params
dataset_name = 'dota';
f = samplerReal(dataset_name);
n_resamples_per_trial = 100;
n_trials_per_corruption_level = 10;  % each trial is subset of total bootstrap, mostly for convergence plots
nprocesses = 12;
n_resamples_per_corruption_level = n_resamples_per_trial*n_trials_per_corruption_level;  % number of bootstrap iterations per corruption level
p0 = 0.1;  % natural corruption (dist1 samples in training)
p_max = 1.0;  % maximum artificial corruption to inject (including p_0)
nV = 100;

% Choose training set size by specifying p_clean_resample
%p_clean_resample = 0.002; % Probability of resampling zero corrupted samples
%nT = floor(log(p_clean_resample)/log(1-p0));  % Number of samples in train set

% Choose training set size directly
nT = 100; %10000;
p_clean_resample = (1-p0)^nT;

n_corruption_levels = 10;%2*nT;

p = linspace(p0, p_max, n_corruption_levels);
B = NaN(length(p), n_trials_per_corruption_level);

pool = gcp('nocreate'); % If no pool, do not create new one.
if isempty(pool)
    poolsize = 0;
else
    poolsize = pool.NumWorkers;
end
if poolsize ~= nprocesses
    if poolsize > 0
        delete(pool)
    end
    parpool(nprocesses);
end

tic;
parfor i = 1:length(p)
    p_i = p(i);
    D = makedist('Binomial','N',nT,'p',p_i);
    A(i,:) = D.pdf(0:nT);
    btrials = zeros(1, n_trials_per_corruption_level);
    for j = 1:n_trials_per_corruption_level
        b_jk = 0;
        for k = 1:n_resamples_per_trial
            n1 = binornd(nT, p_i);
            n0 = nT - n1;
            [x, y, xtest, ytest] = f.sample_dual(n0, nT, 0, nV);
            SVMModel = fitcsvm(x, y);
            yhat = predict(SVMModel, xtest);
            error = 1 - sum(strcmp(yhat, ytest))/length(ytest);
            b_jk = b_jk+error;
        end
        btrials(j) = b_jk/n_resamples_per_trial;
    end
    B(i, :) = btrials;
end
time = toc; tic;
fprintf('B3 sampling time: %f sec \n', time);

% Compute s_true, at most 50 points
s_true_n_corrupted_samples = floor(linspace(0, nT, min(50, nT+1)));
s_true = zeros(length(s_true_n_corrupted_samples), n_trials_per_corruption_level);

parfor i = 1:length(s_true)
    n_corrupted_samples = s_true_n_corrupted_samples(i);
    for j = 1:n_trials_per_corruption_level
        for k = 1:n_resamples_per_corruption_level
            [x, y, xtest, ytest] = f.sample_dual(nT-n_corrupted_samples, nT, 0, nV);
            SVMModel = fitcsvm(x, y);
            yhat = predict(SVMModel, xtest);
            error = 1 - sum(strcmp(yhat, ytest))/length(ytest);
            s_true(i, j) = s_true(i, j)+error/n_resamples_per_corruption_level;
        end
    end
end
t = toc;
fprintf('True error sampling time: %f sec \n', t);

filename = ['AB_', datestr(now, 'yyyy-mm-dd_hh-MM-ss'), '_', git, '_', dataset_name, '_nT', num2str(nT), '_p0', num2str(p0), '_K', num2str(n_resamples_per_corruption_level), '.mat'];
save(filename, 'dataset_name', 'n_corruption_levels', 'n_trials_per_corruption_level', 'n_resamples_per_trial', 'n_resamples_per_corruption_level', 'B', 'nT', 'nV', 's_true', 's_true_n_corrupted_samples', 'p0', 'p', 'nprocesses', 'time', 'notes', 'git')