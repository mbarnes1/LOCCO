clear, clc, close all
% Synthetic experiments
load('AB_2017-02-03_14-43-33.mat');  % Synthetic data for known p0
lambda = 0.1;
% Adult (Census) experiments
%load('AB_2017-03-11_13-21-26.mat');  % Real Adult data 2500
%load('AB_2017-03-17_07-29-51.mat');  % Real Adult data 10000
%lambda = 10;
% Heart experiments
%load('AB_2017-03-24_20-22-09.mat');
%lambda = 10;

b = mean(B, 2);
[cmap, ~, ~] = brewermap(6, 'Set2');
f = figure; hold on

%% Monotonic
s_mono = trendfilter(A, b, 2, 0, true);
plot((0:nT)/nT, s_mono, 'Color', cmap(1,:));

%% Regularize 2nd derivative
s_t2 = trendfilter(A, b, 2, lambda, false);
plot((0:nT)/nT, s_t2, 'Color', cmap(2,:));

%% Regularize 3rd derivative
s_t3 = trendfilter(A, b, 3, lambda, false);
plot((0:nT)/nT, s_t3, 'Color', cmap(3,:));

%% Regularize 2nd derivative, monotonic
s_t2mono = trendfilter(A, b, 2, lambda, true);
plot((0:nT)/nT, s_t2mono, 'Color', cmap(4,:));

%% Regularize 3rd derivative, monotonic
s_t3mono = trendfilter(A, b, 3, lambda, true);
plot((0:nT)/nT, s_t3mono, 'Color', cmap(5,:));

%% Regularize 4th derivative, monotonic
s_t4mono = trendfilter(A, b, 4, lambda, true);
plot((0:nT)/nT, s_t4mono, 'Color', cmap(6,:));

plot((0:nT)/nT, mean(s_true, 2), 'Color', 'k', 'LineStyle', '--');

%% Final result of full error fit
xlabel('Dependency Leakage')
ylabel('Classification Error, e')
legend('mono', 'T2', 'T3', 'T2 + mono', 'T3 + mono', 'T4+mono', 'True')
set(f, 'units', 'inches', 'pos', [0 0 6 4.5])