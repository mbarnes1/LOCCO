clear, clc, close all
% Synthetic experiments
%load('AB_2017-02-03_14-43-33.mat');  % Synthetic data for known p0
load('bootstraps/AB_2018-09-25_17-30-28_synthetic_nT100_p00.1_K100.mat')

lambda = 0.1;
lambda_sketch = 0.01;
sketch_mean = 10;
sketch_block = 10;
polyorder = 7;  % order of polynomial basis function
subsample = 10;

% Adult (Census) experiments
%load('AB_2017-03-11_13-21-26.mat');  % Real Adult data 2500
% load('AB_2017-03-17_07-29-51.mat');  % Real Adult data 10000
% lambda = 10;
% sketch_mean = 6;
% sketch_block = 20;
% lambda_sketch = 0.1;
% polyorder = 7;

% Heart experiments
% load('AB_2017-03-24_20-22-09.mat');
% lambda = 10;
% sketch_mean = 6;
% sketch_block = 20;
% lambda_sketch = 0.1;
% polyorder = 7;

b = mean(B, 2);

%methods = {'mono', 'T2', 'T3', 'T2+mono', 'T3+mono', 'T4+mono', 'Basis', 'Medoid', 'Block', 'True'};
methods = {'T4+mono', 'Basis', 'True'}; %{'T4+mono', 'Basis', 'Sketch', 'True'};
method_counter = 1;
[cmap, ~, ~] = brewermap(length(methods), 'Set2');
f = figure; hold on

%% Monotonic
if any(strcmp(methods,'mono'))
    s_mono = trendfilter(A, b, 2, 0, true, subsample);
    plot((0:nT)/nT, s_mono, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% Regularize 2nd derivative
if any(strcmp(methods, 'T2'))
    s_t2 = trendfilter(A, b, 2, lambda, false, subsample);
    plot((0:nT)/nT, s_t2, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% Regularize 3rd derivative
if any(strcmp(methods, 'T3'))
    s_t3 = trendfilter(A, b, 3, lambda, false, subsample);
    plot((0:nT)/nT, s_t3, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end
%% Regularize 2nd derivative, monotonic
if any(strcmp(methods, 'T2+mono'))
    s_t2mono = trendfilter(A, b, 2, lambda, true, subsample);
    plot((0:nT)/nT, s_t2mono, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% Regularize 3rd derivative, monotonic
if any(strcmp(methods, 'T3+mono'))
    s_t3mono = trendfilter(A, b, 3, lambda, true, subsample);
    plot((0:nT)/nT, s_t3mono, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% Regularize 4th derivative, monotonic
if any(strcmp(methods, 'T4+mono'))
    s_t4mono = trendfilter(A, b, 4, lambda, true, subsample);
    plot((0:nT)/nT, s_t4mono, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% Polynomial basis function
if any(strcmp(methods, 'Basis'))
    s_poly = polyfilter(A, b, polyorder);
    plot((0:nT)/nT, s_poly, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% k-medoid sketching
if any(strcmp(methods, 'Medoid'))
    s_sketch_medoid = filter_sketch(A, b, sketch_medoid, 'medoid', 0, false, subsample);
    plot((0:nT)/nT, s_sketch_medoid, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% block sketching
if any(strcmp(methods, 'Block'))
    s_sketch_block = filter_sketch(A, b, sketch_block, 'block', lambda, true, subsample);
    plot((0:nT)/nT, s_sketch_block, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end


%% mean sketching
if or(any(strcmp(methods, 'Mean')), any(strcmp(methods, 'Sketch')))
    s_sketch_mean = filter_sketch(A, b, sketch_mean, 'mean', lambda, true, subsample);
    plot((0:nT)/nT, s_sketch_mean, 'Color', cmap(method_counter,:));
    method_counter = method_counter + 1;
end

%% true
if any(strcmp(methods, 'True'))
    plot((0:nT)/nT, mean(s_true, 2), 'Color', 'k', 'LineStyle', '--');
end

%% Final result of full error fit
xlabel('Dependency Leakage')
ylabel('Classification Error, e')
legend(methods)
set(f, 'units', 'inches', 'pos', [0 0 6 4.5])