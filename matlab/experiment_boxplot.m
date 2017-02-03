clear, clc, close all
load('AB_2016-12-05_15-42-47.mat');

results = NaN(trials, 7);

lambda = 0.1;
for trial = 1:trials
    b = B(:, trial);
    
    %% Naive
    x_naive = b(1);
    
    %% Linear regression (this does terribly)
    %x_linear = (A'*A) \ (A' * b);
    %x_linear = x_linear(1);

    %% Ridge Regression
    %x_ridge = (A'*A + lambda * eye(size(A, 2))) \ (A' * b);
    %x_ridge = x_ridge(1);
    
    %% Monotonic, linear
    x_mono = trendfilter(A, b, 2, 0, true);
    x_mono = x_mono(1);
    
    %% Regularize 2nd derivative
    x_trend2 = trendfilter(A, b, 2, lambda, false);
    x_trend2 = x_trend2(1);
    
    %% Regularize 3rd derivative
    x_trend3 = trendfilter(A, b, 3, lambda, true);
    x_trend3 = x_trend3(1);
    
    %% Regularize 2nd derivative, monotonic
    x_trend2mono = trendfilter(A, b, 2, lambda, true);
    x_trend2mono = x_trend2mono(1);
    
    %% Regularize 3rd derivative, monotonic
    x_trend3mono = trendfilter(A, b, 3, lambda, true);
    x_trend3mono = x_trend3mono(1);
    
    %% Regularize 4th derivative, monotonic
    x_trend4mono = trendfilter(A, b, 4, lambda, true);
    x_trend4mono = x_trend4mono(1);
    
    results(trial, :) = [x_naive, x_mono, x_trend2, x_trend2mono, x_trend3, x_trend3mono, x_trend4mono];
end

%% Final result of full error fit
f = figure;
[cmap, ~, ~] = brewermap(3, 'Set2');
boxplot(abs(results - s_true(1)), {'Base', 'Mono', 'T2', 'T2+mono', 'T3', 'T3+mono', 'T4+mono'});
h_median = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(h_median, 'Color', cmap(2,:));
h_box = findobj('Tag','Box');
set(h_box, 'Color', cmap(3,:));
ylabel('Absolute error, $$|\hat x_0 - x_0|$$', 'Interpreter', 'latex')
ylim([0 inf])
set(f, 'units', 'inches', 'pos', [0 0 5.5 4.125])