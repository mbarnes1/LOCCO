clear, clc, close all
load('AB.mat')

S = NaN(3, steps);
for i = 1:steps
    b = B(:, i);
    
    %% Ridge Regression
    lambda = .1;
    s_ridge = (A'*A + lambda * eye(size(A, 2))) \ (A' * b);
    S(1, i) = s_ridge(1);

    %% Regularize 2nd derivative, monotonic
    lambda = 0.01;
    pos_mono = eye(nT+1) - diag(ones(nT,1), 1);
    trend = eye(nT+1) - 2*diag(ones(nT,1), 1) + diag(ones(nT-1,1), 2);
    cvx_begin
        variable s(nT+1)
        minimize( norm(A*s - b, 2) + lambda * norm(trend*s, 2) )
        subject to
            pos_mono*s >= 0;
    cvx_end
    s_trend2 = s;
    S(2, i) = s_trend2(1);
    
    %% Regularize 3rd derivative, monotonic
    lambda = 0.01;
    pos_mono = eye(nT+1) - diag(ones(nT,1), 1);
    trend = - eye(nT+1) + 3*diag(ones(nT,1), 1) - 3*diag(ones(nT-1,1), 2) + diag(ones(nT-2,1), 3);
    cvx_begin
        variable s(nT+1)
        minimize( norm(A*s - b, 2) + lambda * norm(trend*s, 2) )
        subject to
            pos_mono*s >= 0;
    cvx_end
    s_trend3 = s;
    S(2, i) = s_trend3(1);
end
plot((0:nT)/nT, s_ridge,...
    (0:nT)/nT, s_trend2,...
    (0:nT)/nT, s_trend3,...
    (0:nT)/nT, s_true, '--')
xlabel('Corruption')
ylabel('MSE')
legend('Ridge', 'Trend, 2nd', 'Trend, 3rd', 'True')

figure
plot(savestep*(1:steps), S)
xlabel('Bootstrap samples (t)')
ylabel('x_0')
legend('Ridge', 'Trend, 2nd', 'Trend, 3rd')