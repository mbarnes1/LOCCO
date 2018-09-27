clear, clc, close all

% Params
dataset_name = 'synthetic';
n_resamples_per_trial = 10;
n_trials_per_corruption_level = 10;
nprocesses = 2;
n_resamples_per_corruption_level = n_resamples_per_trial*n_trials_per_corruption_level;  % number of bootstrap iterations per corruption level
p0 = 0.1;  % natural corruption (dist1 samples in training)
nV = 1000;  % Size of validation set

% Choose training set size by specifying p_clean_resample
%p_clean_resample = 0.002; % Probability of resampling zero corrupted samples
%nT = floor(log(p_clean_resample)/log(1-p0));  % Number of samples in train set

% Choose training set size directly
nT = 10; %10000;
p_clean_resample = (1-p0)^nT;

n_corruption_levels = 2*nT;

% Slopes
m0 = 1;
m1 = 2;
sigma0 = 0.5;
sigma1 = 0.5;
range0 = [-1, 1];
range1 = [-10, 10];
f = sampler(m0, m1, sigma0, sigma1, range0, range1);

p = linspace(p0, 1, n_corruption_levels);
%A = NaN(length(p), nT+1);
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

%% B3 resamples
tic
parfor i = 1:length(p)
    p_i = p(i);
    D = makedist('Binomial','N',nT,'p',p_i);
    %A(i,:) = D.pdf(0:nT);
    btrials = zeros(1, n_trials_per_corruption_level);
    for j = 1:n_trials_per_corruption_level
        b_jk = 0;
        for k = 1:n_resamples_per_trial
            n1 = binornd(nT, p_i);  % number of corrupted samples
            n0 = nT - n1;  % number of uncorrupted samples
            [x, y] = f.sample(n0, nT);
            z = dot(x,y)/(dot(x, x));
            [xtest, ytest] = f.sample(0, nV);
            yhat = xtest*z;
            mse = mean((ytest-yhat).^2);
            b_jk = b_jk+mse;
        end
        btrials(j) = b_jk/n_resamples_per_trial;
    end
    B(i, :) = btrials;
end
time = toc;

%% Estimate s_true
s_true = zeros(nT+1, n_trials_per_corruption_level);

parfor i = 1:length(s_true)
    for j = 1:n_trials_per_corruption_level
        for k = 1:n_resamples_per_corruption_level
            [x, y] = f.sample(nT-i+1, nT);
            z = dot(x,y)/(dot(x, x));
            [xtest, ytest] = f.sample(0, nV);
            yhat = xtest*z;
            mse = mean((ytest-yhat).^2);
            s_true(i, j) = s_true(i, j)+mse/n_resamples_per_corruption_level;
        end
    end
end

%% Save results
filename = ['AB_', datestr(now, 'yyyy-mm-dd_hh-MM-ss'), '_', dataset_name, '_nT', num2str(nT), '_p0', num2str(p0), '_K', num2str(n_resamples_per_corruption_level), '.mat'];
save(filename, 'dataset_name', 'n_corruption_levels', 'n_trials_per_corruption_level', 'n_resamples_per_trial', 'n_resamples_per_corruption_level', 'B', 'nT', 'nV', 's_true', 'p0', 'p', 'm0', 'm1', 'sigma0', 'sigma1', 'range0', 'range1', 'nprocesses', 'time')