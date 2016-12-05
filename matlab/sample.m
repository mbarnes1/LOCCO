clear, clc, close all

% Params
npertrial = 10;
trials = 5;
nprocesses = 2;
K = npertrial*trials;  % number of bootstrap iterations per corruption level
p0 = 0.1;  % natural corruption (dist1 samples in training)
p_clean_resample = 0.2; % Probability of resampling zero corrupted samples
nT = floor(log(p_clean_resample)/log(1-p0));  % Number of samples in train set
nV = 1000;  % Size of validation set

% Slopes
m0 = 1;
m1 = 2;
sigma0 = 0.5;
sigma1 = 0.5;
f = sampler(m0, m1, sigma0, sigma1);

p = linspace(p0, 1, 2*nT);
A = NaN(length(p), nT+1);
B = NaN(length(p), trials);

parpool(nprocesses);
parfor i = 1:length(p)
    p_i = p(i);
    btrials = NaN(trials, 1);
    D = makedist('Binomial','N',nT,'p',p_i);
    A(i,:) = D.pdf(0:nT);
    btrials = zeros(1, trials);
    for j = 1:trials
        b_jk = 0;
        for k = 1:npertrial
            n1 = binornd(nT, p_i);
            n0 = nT - n1;
            [x, y] = f.sample(n0, nT);
            z = dot(x,y)/(dot(x, x));
            [xtest, ytest] = f.sample(0, nV);
            yhat = xtest*z;
            mse = mean((ytest-yhat).^2);
            b_jk = b_jk+mse;
        end
        btrials(j) = b_jk/npertrial;
    end
    B(i, :) = btrials;
end

s_true = zeros(nT+1, 1);
for i = 1:length(s_true)
    for j = 1:K
        [x, y] = f.sample(nT-i+1, nT);
        z = dot(x,y)/(dot(x, x));
        [xtest, ytest] = f.sample(0, nV);
        yhat = xtest*z;
        mse = mean((ytest-yhat).^2);
        s_true(i) = s_true(i)+mse/K;
    end
end

filename = ['AB_', datestr(now, 'yyyy-mm-dd_hh-MM-ss'), '.mat'];
save(filename, 'trials', 'B', 'A', 'nT', 'nV', 'K', 's_true', 'npertrial', 'p0', 'm0', 'm1', 'sigma0', 'sigma1')