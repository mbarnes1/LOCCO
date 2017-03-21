% Preliminary analysis of whether a dataset is appropriate for LOCO
clear, clc, close all
f = samplerReal('heart');

% Naive cross-validation
K = 100;
ntrials = 10;
naiveloss = NaN(ntrials, 1);
locoloss = NaN(ntrials, 1);
for trial = 1:ntrials
    classLoss = 0;
    for k = 1:K
        %fprintf('    naive fold %i \n', k);
        [x, y, xtest, ytest] = f.sample_dual(50, 100, 0, 100);
        SVMModel = fitcsvm(x, y);
        yhat = predict(SVMModel, xtest);
        error = 1 - sum(strcmp(yhat, ytest))/length(ytest);
        classLoss = classLoss + error/K;
    end
    naiveloss(trial) = classLoss;
    fprintf('Naive cross-validation error: %f \n', classLoss);

    % LOCO
    classLoss = 0;
    for k = 1:K
        %fprintf('    naive fold %i \n', k);
        [x, y, xtest, ytest] = f.sample_dual(100, 100, 0, 100);
        SVMModel = fitcsvm(x, y);
        yhat = predict(SVMModel, xtest);
        error = 1 - sum(strcmp(yhat, ytest))/length(ytest);
        classLoss = classLoss + error/K;
    end
    locoloss(trial) = classLoss;
    fprintf('LOCO cross-validation error: %f \n', classLoss);

    %disp('Per class errors:')
    %disp(results)
end

boxplot([naiveloss, locoloss], {'Naive', 'LOCO'})
ylabel('Classification Error')

