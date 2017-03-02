% Preliminary analysis of whether a dataset is appropriate for LOCO
dataset = 'data/adult.csv';
T = readtable(dataset, 'Delimiter',',');

featuresandlabel = {'age', 'education_num', 'hours_per_week', 'race', 'income'};
clustername = {'native_country'};
clusters_to_consider = {'United-States', 'Canada', 'El-Salvador', 'Germany', 'India', 'Mexico', 'Philippines', 'Puerto-Rico'};
labelname = 'income';
ntrain = 10;

clusters = table2cell(T(:, clustername));
ind = ismember(clusters, clusters_to_consider);
clusters = clusters(ind);
X = T(ind, featuresandlabel);


% Naive cross-validation
K = 10;
cvind = crossvalind('Kfold', length(clusters), K);
classLoss = 0;
for k = 1:K
    fprintf('    naive fold %i \n', k);
    test_ind = cvind==k;
    train_ind = randsample(find(test_ind == 0), ntrain);
    SVMModel = fitcsvm(X(train_ind, :), labelname);
    classLoss = classLoss + sum(strcmp(predict(SVMModel, X(test_ind,:)), table2cell(X(test_ind, labelname))));
end
classLoss = 1 - classLoss/length(clusters);
sprintf('Naive cross-validation error: %f', classLoss)

% LOCO
categories = unique(clusters);
totalmistakes = 0;
results = table(NaN(length(categories), 1), 'RowNames', categories, 'VariableNames', {'Test_error'});
for i = 1:length(categories)
    test_cluster = categories{i};
    fprintf('    LOCO on test class %s \n', test_cluster);
    test_ind = strcmp(clusters, test_cluster);
    train_ind = randsample(find(test_ind == 0), ntrain);
    SVMModel = fitcsvm(X(train_ind, :), labelname);
    mistakes = sum(test_ind) - sum(strcmp(predict(SVMModel, X(test_ind,:)), table2cell(X(test_ind, labelname))));
    results{test_cluster, 'Test_error'} = mistakes/sum(test_ind);
    totalmistakes = totalmistakes + mistakes;
end
classLoss = totalmistakes/length(test_ind);
fprintf('LOCO cross-validation error: %f \n', classLoss)
disp('Per class errors:')
disp(results)


