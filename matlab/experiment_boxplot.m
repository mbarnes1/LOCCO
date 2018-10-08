clear, clc
% Synthetic experiments
%load('AB_2017-02-03_14-43-33.mat');  % Synthetic data for known p0
% lambda = 0.1;
% lambda_sketch = 0.01;
% sketch_mean = 10;
% sketch_block = 10;
% polyorder = 7;

% Real experiments
%load('AB_2017-03-11_13-21-26.mat');  % Real Adult data 2500
%load('AB_2017-03-17_07-29-51.mat');  % Real Adult data 10000
%lambda = 10;
%sketch_mean = 6;
%sketch_block = 20;
%lambda_sketch = 0.1;
%polyorder = 7;
% Heart experiments
%load('AB_2017-03-24_20-22-09.mat');
%lambda = 10;
%polyorder = 7;

% Parkinson
load('bootstraps/parkinson/AB_2018-10-02_21-32-21_62c9b1_parkinson_nT100_p00.1_K10000.mat');
lambda = 1000;
sketch_mean = 6;
sketch_block = 20;
lambda_sketch = 0.1;
polyorder = 2;
subsample = 1;


% Dota 2
% load('bootstraps/dota/AB_2018-10-02_21-22-56_62c9b1_dota_nT1000_p00.1_K10000.mat');
% lambda = 1000;
% sketch_mean = 6;
% sketch_block = 20;
% lambda_sketch = 0.1;
% polyorder = 2;
% subsample = 1;

% Note basis = basis6
methods = {'IID', 'LOCO', 'T4+mono', 'Basis', 'Sketch'}; %'T4+mono', 'Basis', 'Sketch'};
%methods = {'IID', 'LOCO', 'Mono', 'T2', 'T2+mono', 'T3', 'T3+mono', 'T4+mono', 'Basis', 'Medoid', 'Block', 'Mean'};
results = NaN(n_trials_per_corruption_level, length(methods));

%% Compute A
A = NaN(length(p), nT+1);
for i = 1:length(p)
    p_i = p(i);
    D = makedist('Binomial','N',nT,'p',p_i);
    A(i,:) = D.pdf(0:nT);
end

%% Run trials
for trial = 1:n_trials_per_corruption_level
    b = B(:, trial);
    method_counter = 1;
    
    %% IID
    if any(strcmp(methods,'IID'))
        if strcmp(dataset_name, 'parkinson') 
            x_iid = b(find(p > 0.5, 1));  % full s_true not available
        else
            x_iid = s_true(round(size(s_true,1)/2), trial);
        end
        
        results(trial, method_counter) = x_iid;
        method_counter = method_counter + 1;
    end
    
    %% LOCO
    if any(strcmp(methods,'LOCO'))
        x_loco = b(1);
        results(trial, method_counter) = x_loco;
        method_counter = method_counter + 1;
    end
    
    %% Monotonic, linear
    if any(strcmp(methods,'Mono'))
        x_mono = trendfilter(A, b, 2, 0, true);
        x_mono = x_mono(1);
        results(trial, method_counter) = x_mono;
        method_counter = method_counter + 1;
    end
    
    %% Regularize 2nd derivative
    if any(strcmp(methods,'T2'))
        x_trend2 = trendfilter(A, b, 2, lambda, false);
        x_trend2 = x_trend2(1);
        results(trial, method_counter) = x_trend2;
        method_counter = method_counter + 1;
    end
    
    %% Regularize 3rd derivative
    if any(strcmp(methods,'T3'))
        x_trend3 = trendfilter(A, b, 3, lambda, true);
        x_trend3 = x_trend3(1);
        results(trial, method_counter) = x_trend3;
        method_counter = method_counter + 1;
    end
    
    
    %% Regularize 2nd derivative, monotonic
    if any(strcmp(methods,'T2+mono'))
        x_trend2mono = trendfilter(A, b, 2, lambda, true);
        x_trend2mono = x_trend2mono(1);
        results(trial, method_counter) = x_trend2mono;
        method_counter = method_counter + 1;
    end
    
    
    %% Regularize 3rd derivative, monotonic
    if any(strcmp(methods,'T3+mono'))
        x_trend3mono = trendfilter(A, b, 3, lambda, true);
        x_trend3mono = x_trend3mono(1);
        results(trial, method_counter) = x_trend3mono;
        method_counter = method_counter + 1;
    end
    
    %% Regularize 4th derivative, monotonic
    if any(strcmp(methods,'T4+mono'))
        x_trend4mono = trendfilter(A, b, 4, lambda, true, subsample);
        x_trend4mono = x_trend4mono(1);
        results(trial, method_counter) = x_trend4mono;
        method_counter = method_counter + 1;
    end
    
    %% Basisnomial basis
    if any(strcmp(methods,'Basis'))
        x_poly = polyfilter(A, b, polyorder);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis1'))
        x_poly = polyfilter(A, b, 1);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis2'))
        x_poly = polyfilter(A, b, 2);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis3'))
        x_poly = polyfilter(A, b, 3);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis4'))
        x_poly = polyfilter(A, b, 4);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis5'))
        polyorder = 5;
        x_poly = polyfilter(A, b, 5);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis6'))
        polyorder = 6;
        x_poly = polyfilter(A, b, 6);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis7'))
        x_poly = polyfilter(A, b, 7);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis8'))
        x_poly = polyfilter(A, b, 8);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis9'))
        x_poly = polyfilter(A, b, 9);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    if any(strcmp(methods,'Basis10'))
        x_poly = polyfilter(A, b, 10);
        x_poly = x_poly(1);
        results(trial, method_counter) = x_poly;
        method_counter = method_counter + 1;
    end
    
    %% Sketching -- medoid
    if any(strcmp(methods,'Medoid'))
        x_sketch_medoid = filter_sketch(A, b, sketch_medoid, 'medoid', 0, false);
        results(trial, method_counter) = x_sketch_medoid(1);
        method_counter = method_counter + 1;
    end
    
    %% Sketching -- blocking
    if any(strcmp(methods,'Block'))
        x_sketch_block = filter_sketch(A, b, sketch_block, 'block', lambda_sketch, true);
        results(trial, method_counter) = x_sketch_block(1);
        method_counter = method_counter + 1;
    end
    
    %% Sketching -- mean
    if or(any(strcmp(methods,'Mean')), any(strcmp(methods,'Sketch')))
        x_sketch_mean = filter_sketch(A, b, sketch_mean, 'mean', lambda_sketch, true, subsample);
        results(trial, method_counter) = x_sketch_mean(1);
        method_counter = method_counter + 1;
    end
end

%% Final result of full error fit
f = figure;
%plot([0.5, 7.5],[0 0], 'Color', 0.75 * ones(1, 3));
hold on;
[cmap, ~, ~] = brewermap(3, 'Set2');
boxplot(abs(mean(s_true(1, :)) - results), methods, 'Whisker', 99);
h_median = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(h_median, 'Color', cmap(2,:));
h_box = findobj('Tag','Box');
set(h_box, 'Color', cmap(3,:));
ylabel('Absolute Error, $$|e_0 - \hat e_0|$$', 'Interpreter', 'latex')
%ylim([0 1.05*max(max(abs(results - mean(s_true(1, :)))))])
% Do this last, once methods are finalized
%h_xlab = xlabel('Baselines   |                                 Our method');
%set(h_xlab,'Position',get(h_xlab,'Position') - [0.8 0 0])
%set(h_xlab,'Position',[3.7 -0.0011 0])
set(f, 'units', 'inches', 'pos', [0 0 6 4.5])