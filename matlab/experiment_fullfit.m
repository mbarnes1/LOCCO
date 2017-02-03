clear, clc, close all
%load('AB_2016-12-05_13-56-12.mat')
load('AB_2016-12-05_15-42-47.mat')
[cmap, ~, ~] = brewermap(5, 'Set2');

b = mean(B, 2);

lambda = 0.1;

figure; hold on
%% Regularize 2nd derivative
s_t2 = trendfilter(A, b, 2, lambda, false);
plot((0:nT)/nT, s_t2, 'Color', cmap(1,:));

%% Regularize 3rd derivative
s_t3 = trendfilter(A, b, 3, lambda, false);
plot((0:nT)/nT, s_t3, 'Color', cmap(2,:));
%% Regularize 2nd derivative
s_t2mono = trendfilter(A, b, 2, lambda, true);
plot((0:nT)/nT, s_t2mono, 'Color', cmap(3,:));
%% Regularize 3rd derivative
s_t3mono = trendfilter(A, b, 3, lambda, true);
plot((0:nT)/nT, s_t3mono, 'Color', cmap(4,:));
%% Regularize 4th derivative, monotonic
s_t4mono = trendfilter(A, b, 4, lambda, true);
plot((0:nT)/nT, s_t4mono, 'Color', cmap(5,:));

plot((0:nT)/nT, s_true, 'Color', 'k', 'LineStyle', '--');

%% Final result of full error fit
%f = figure;
%plot((0:nT)/nT, s_t2,...
%    (0:nT)/nT, s_t3,...
%    (0:nT)/nT, s_t2mono,...
%    (0:nT)/nT, s_t3mono,...
%    (0:nT)/nT, s_t4mono,...
%    (0:nT)/nT, s_true, '--')
xlabel('Corruption')
ylabel('MSE')
legend('T2', 'T3', 'T2 + mono', 'T3 + mono', 'T4+mono', 'True')