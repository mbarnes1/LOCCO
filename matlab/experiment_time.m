clear, clc, close all
cvx_quiet true

% Synthetic experiments
% load('AB_2017-02-03_14-43-33.mat');  % Synthetic data for known p0
% lambda = 0.1;
% lambda_sketch = 0.01;
% sketch_mean = 10;
% sketch_block = 10;
% polyorder = 7;
% Adult (Census) experiments
%load('AB_2017-03-11_13-21-26.mat');  % Real Adult data 2500
%load('AB_2017-03-17_07-29-51.mat');  % Real Adult data 10000
lambda = 10;
sketch_mean = 6;
sketch_block = 20;
lambda_sketch = 0.1;
polyorder = 7;
% Heart experiments
load('AB_2017-03-24_20-22-09.mat');
%lambda = 10;

b = mean(B, 2);

%% Monotonic
% f = @() trendfilter(A, b, 2, 0, true);
% t_mono = timeit(f);
% fprintf('trend filter, mono: %.8fs per call\n', t_mono)

% %% Regularize 2nd derivative
% f = @() trendfilter(A, b, 2, lambda, false);
% t_t2 = timeit(f);
% 
% %% Regularize 3rd derivative
% f = @() trendfilter(A, b, 3, lambda, false);
% t_t3 = timeit(f);
% 
% %% Regularize 2nd derivative, monotonic
% f = @() trendfilter(A, b, 2, lambda, true);
% t_t2mono = timeit(f);
% 
% %% Regularize 3rd derivative, monotonic
% f = @() trendfilter(A, b, 3, lambda, true);
% t_t3mono = timeit(f);
% 

%% Regularize 4th derivative, monotonic
f = @() trendfilter(A, b, 4, lambda, true);
t_t4mono = timeit(f);
fprintf('4th order trend filter + mono: %.8fs per call\n', t_t4mono)


%% Polynomial basis function
f = @() polyfilter(A, b, polyorder);
t_poly = timeit(f);
fprintf('Poly basis: %.8fs per call\n', t_poly)

%% Mean sketching
f = @() filter_sketch(A, b, sketch_mean, 'mean', lambda_sketch, true);
t_poly = timeit(f);
fprintf('Mean sketch: %.8fs per call\n', t_poly)


