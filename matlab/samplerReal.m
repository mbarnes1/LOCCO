classdef samplerReal < handle
    %SAMPLER Sample from some mix of the training and testing distributions
    
    properties (SetAccess = private)
        dataset
        features
        clustername
        train_clusters
        test_clusters
        label
        X0
        Y0
        X1
        Y1
    end

    methods
        function self = samplerReal(dataset)
            if dataset == 'adult'
                self.dataset = 'data/adult.csv';
                self.features = {'age', 'education_num', 'hours_per_week', 'race', 'occupation'};
                self.clustername = {'native_country'};
                self.train_clusters = {'United-States', 'El-Salvador', 'Germany', 'Mexico', 'Philippines', 'Puerto-Rico'};
                self.test_clusters = {'India', 'Canada'};
                self.label = 'income';
            elseif dataset == 'heart'
                self.dataset = 'data/heart/heart_all.csv';
                self.features = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'thal'};
                self.clustername = {'country'};
                self.train_clusters = {'cleveland', 'va', 'switzerland'};
                self.test_clusters = {'hungarian'};
                self.label = 'num';
            else
                error('Invalid dataset name')
            end
            T = readtable(self.dataset, 'Delimiter',',');
            clusters = table2cell(T(:, self.clustername));
            self.X0 = T(ismember(clusters, self.train_clusters), self.features);
            self.Y0 = T(ismember(clusters, self.train_clusters), self.label);
            self.X1 = T(ismember(clusters, self.test_clusters), self.features);
            self.Y1 = T(ismember(clusters, self.test_clusters), self.label);
        end

        function [x, y] = sample(self, n0, n)
            % n0: Number of training samples
            % n: Total number of samples
            n1 = n - n0;
            ind0 = randsample(height(self.Y0), n0);
            ind1 = randsample(height(self.Y1), n1);
            x = vertcat(self.X0(ind0, :), self.X1(ind1, :));
            y = table2cell(vertcat(self.Y0(ind0, :), self.Y1(ind1, :)));
        end
        
        function [xa, ya, xb, yb] = sample_dual(self, n0a, na, n0b, nb)
            % Samples such that no overlap between set a and b
            % n0a: Number of training samples in set A
            % na: Total number of samples in set A
            % n0b: Number of training samples in set A
            % nb: Total number of samples in set A
            n1a = na - n0a;
            n1b = nb - n0b;
            ind0 = randsample(height(self.Y0), n0a + n0b);
            ind1 = randsample(height(self.Y1), n1a + n1b);
            xa = vertcat(self.X0(ind0(1:n0a), :), self.X1(ind1(1:n1a), :));
            xb = vertcat(self.X0(ind0((n0a+1):end), :), self.X1(ind1((n1a+1):end), :));
            ya = table2cell(vertcat(self.Y0(ind0(1:n0a), :), self.Y1(ind1(1:n1a), :)));
            yb = table2cell(vertcat(self.Y0(ind0((n0a+1):end), :), self.Y1(ind1((n1a+1):end), :)));
        end
    end
    
end

