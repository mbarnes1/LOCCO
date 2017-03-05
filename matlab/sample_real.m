clear, clc, close all

% Params
npertrial = 100;
trials = 10;  % each trial is subset of total bootstrap, mostly for convergence plots
nprocesses = 32;
K = npertrial*trials;  % number of bootstrap iterations per corruption level
p0 = 0.1;  % natural corruption (dist1 samples in training)
nT = 100;  % Number of samples in train set
nV = 100;

% Slopes
f = samplerReal();

p = linspace(p0, 1, 2*nT);
A = NaN(length(p), nT+1);
B = NaN(length(p), trials);

parpool(nprocesses);
tic;
parfor i = 1:length(p)
    p_i = p(i);
    D = makedist('Binomial','N',nT,'p',p_i);
    A(i,:) = D.pdf(0:nT);
    btrials = zeros(1, trials);
    for j = 1:trials
        b_jk = 0;
        for k = 1:npertrial
            n1 = binornd(nT, p_i);
            n0 = nT - n1;
            [x, y, xtest, ytest] = f.sample_dual(n0, nT, 0, nV);
            SVMModel = fitcsvm(x, y);
            yhat = predict(SVMModel, xtest);
            error = 1 - sum(strcmp(yhat, ytest))/length(ytest);
            b_jk = b_jk+error;
        end
        btrials(j) = b_jk/npertrial;
    end
    B(i, :) = btrials;
end
t = toc; tic;
fprintf('B3 sampling time: %f sec \n', t);

s_true = zeros(nT+1, trials);

parfor i = 1:length(s_true)
    for j = 1:trials
        for k = 1:K
            [x, y, xtest, ytest] = f.sample_dual(nT-i+1, nT, 0, nV);
            SVMModel = fitcsvm(x, y);
            yhat = predict(SVMModel, xtest);
            error = 1 - sum(strcmp(yhat, ytest))/length(ytest);
            s_true(i, j) = s_true(i, j)+error/K;
        end
    end
end
t = toc;
fprintf('True error sampling time: %f sec \n', t);

filename = ['AB_', datestr(now, 'yyyy-mm-dd_hh-MM-ss'), '.mat'];
save(filename, 'trials', 'B', 'A', 'nT', 'nV', 'K', 's_true', 'npertrial', 'p0', 'f')