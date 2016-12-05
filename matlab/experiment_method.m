clear, clc, close all
load('AB_2016-12-05_13-56-12.mat')

%% Method convergence comparison
S = NaN(4, steps);
for i = 1:steps
    b = mean(B(:, 1:i), 2);
    
    %% Linear regression
    s_linear = (A'*A) \ (A' * b);
    S(1, i) = s_linear(1);
    
    %% Ridge Regression
    lambda = .1;
    s_ridge = (A'*A + lambda * eye(size(A, 2))) \ (A' * b);
    S(2, i) = s_ridge(1);

    %% Regularize 2nd derivative, monotonic
    lambda = 0.01;
    s_trend2 = trendfilter(A, b, 2, lambda);
    S(3, i) = s_trend2(1);
    
    %% Regularize 3rd derivative, monotonic
    lambda = 0.01;
    s_trend3 = trendfilter(A, b, 3, lambda);
    S(4, i) = s_trend3(1);
end
figure
plot(savestep*(1:steps), abs(S-s_true(1)))
xlabel('Bootstrap samples (t)')
ylabel('Absolute Error')
legend('Linear', 'Ridge', 'Trend, 2nd', 'Trend, 3rd')