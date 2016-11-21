clear, clc, close all

% Params
savestep = 100;
steps = 20;
nprocesses = 2;
K = savestep*steps;  % number of bootstrap iterations per corruption level
p0 = 0.1;  % natural corruption (dist1 samples in training)
p_clean_resample = 0.2; % Probability of resampling zero corrupted samples
nT = floor(log(p_clean_resample)/log(1-p0));  % Number of samples in train set
nV = 1000;  % Size of validation set

% Slopes
m0 = 1;
m1 = 1.5;
sigma0 = 0.5;
sigma1 = 0.5;
f = sampler(m0, m1, sigma0, sigma1);

p = linspace(p0, 1, 2*nT);
A = NaN(length(p), nT+1);
B = NaN(length(p), steps);
b = zeros(length(p), 1);

parpool(nprocesses);
parfor i = 1:length(p)
    pi = p(i);
    bsteps = NaN(steps, 1);
    D = makedist('Binomial','N',nT,'p',pi);
    A(i,:) = D.pdf(0:nT);
    for j = 1:K
        n1 = binornd(nT, pi);
        n0 = nT - n1;
        [x, y] = f.sample(n0, nT);
        z = dot(x,y)/(dot(x, x));
        [xtest, ytest] = f.sample(1.0, nV);
        yhat = xtest*z;
        mse = mean((ytest-yhat).^2);
        b(i) = b(i)+mse;
        if mod(j, savestep) == 0  % save results up to current time
            bsteps(j/savestep) = b(i)/j;
        end
    end
    B(i, :) = bsteps;
end

s_true = zeros(nT+1, 1);
for i = 1:length(s_true)
    for j = 1:K
        [x, y] = f.sample(nT-i+1, nT);
        z = dot(x,y)/(dot(x, x));
        [xtest, ytest] = f.sample(1.0, nV);
        yhat = xtest*z;
        mse = mean((ytest-yhat).^2);
        s_true(i) = s_true(i)+mse/K;
    end
end

save('AB.mat', 'steps', 'B', 'A', 'nT', 'nV', 'K', 's_true', 'savestep')