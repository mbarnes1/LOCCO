clear, clc, close all

% Params
savestep = 100;
steps = 20;
nprocesses = 2;
K = savestep*steps;  % number of bootstrap iterations per corruption level
p0 = 0.1;  % natural corruption (dist1 samples in training)
p_clean_resample = 0.2; % Probability of resampling zero corrupted samples
n = 500;

% Slopes
m0 = 1;
m1 = 2;
sigma0 = 0.5;
sigma1 = 0.5;
range0 = [-1, 1];
range1 = [-1, 1];
f = sampler(m0, m1, sigma0, sigma1, range0, range1);


[x, y] = f.sample(n, n);
[xtest, ytest] = f.sample(0, n);
groups = [zeros(length(x), 1); ones(length(xtest), 1)];
[cmap, ~, ~] = brewermap(2, 'Set2');
gscatter([x; xtest], [y; ytest], groups, cmap,'..', 15)
h = legend('$$\mathcal{T}$$', '$$\mathcal{V}$$', 'Location', 'NorthWest');
set(h,'Interpreter','latex')
xlim([-1, 1])
xlabel('$$x_0$$', 'Interpreter', 'latex')
ylabel('$$x_1$$', 'Interpreter', 'latex')
%title('Uncorrupted Samples')