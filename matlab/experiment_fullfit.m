clear, clc, close all
load('AB_2016-12-05_13-56-12.mat')

b = mean(B, 2);

%% Ridge Regression
lambda = .1;
s_ridge = (A'*A + lambda * eye(size(A, 2))) \ (A' * b);

%% Regularize 2nd derivative, monotonic
lambda = 0.01;
s_trend2 = trendfilter(A, b, 2, lambda);

%% Regularize 3rd derivative, monotonic
lambda = 0.01;
s_trend3 = trendfilter(A, b, 3, lambda);

%% Final result of full error fit
figure
plot((0:nT)/nT, s_ridge,...
    (0:nT)/nT, s_trend2,...
    (0:nT)/nT, s_trend3,...
    (0:nT)/nT, s_true, '--')
xlabel('Corruption')
ylabel('MSE')
legend('Ridge', 'Trend, 2nd', 'Trend, 3rd', 'True')