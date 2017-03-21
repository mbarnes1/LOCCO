clear, clc, close all
% Synthetic experiments
%load('AB_2017-02-03_14-43-33.mat');  % Synthetic data for known p0
%lambda = 0.1;
% Real experiments
%load('AB_2017-03-11_13-21-26.mat');  % Real Adult data 2500
load('AB_2017-03-17_07-29-51.mat');  % Real Adult data 10000
lambda = 100;

results = NaN(trials, 8);

for trial = 1:trials
    b = B(:, trial);
    
    %% IID
    x_iid = s_true(round(size(s_true,1)/2), trial);
    
    %% LOCO
    x_loco = b(1);
    
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
    
    results(trial, :) = [x_iid, x_loco, x_mono, x_trend2, x_trend2mono, x_trend3, x_trend3mono, x_trend4mono];
end

%% Final result of full error fit
f = figure;
[cmap, ~, ~] = brewermap(3, 'Set2');
boxplot(abs(results - mean(s_true(1, :))), {'IID', 'LOCO', 'Mono', 'T2', 'T2+mono', 'T3', 'T3+mono', 'T4+mono'}, 'Whisker', 99);
h_median = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(h_median, 'Color', cmap(2,:));
h_box = findobj('Tag','Box');
set(h_box, 'Color', cmap(3,:));
ylabel('Absolute error, $$|\hat e_0 - e_0|$$', 'Interpreter', 'latex')
ylim([0 1.05*max(max(abs(results - mean(s_true(1, :)))))])
h_xlab = xlabel('Baselines   |                                 Our method');
%set(h_xlab,'Position',get(h_xlab,'Position') - [0.8 0 0])
set(h_xlab,'Position',[3.7 -0.001 0])
set(f, 'units', 'inches', 'pos', [0 0 6 4.5])