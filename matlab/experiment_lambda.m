clear, clc, close all
load('AB10.mat')

%% Regularization convergence experiment
nlambdas = 5;
S = NaN(nlambdas, steps);
methods = cell(nlambdas,1);
order = 3;
lambdas = logspace(-4, 0, nlambdas);

f = @(lambda) (@(b) trendfilter(A, b, order, lambda));
for i = 1:nlambdas
    lambda = lambdas(i);
    methods{i} = f(lambda);
end

for i = 1:steps
    b = B(:, i);
    for j = 1:length(methods)
        method = methods{j};
        s = method(b);
        S(j, i) = s(1);
    end
end
figure
plot(savestep*(1:steps), abs(S-s_true(1)))
xlabel('Bootstrap samples (t)')
ylabel('Absolute Error')
legend(strread(num2str(lambdas),'%s'))
title('Trend Filter Regularization');
