classdef samplerReal < handle
    %SAMPLER Sample from some mix of the training and testing distributions
    
    properties (SetAccess = private)
        dataset = 'data/adult.csv';
        features = {'age', 'education_num', 'hours_per_week', 'race', 'occupation'};
        clustername = {'native_country'};
        train_clusters = {'United-States', 'El-Salvador', 'Germany', 'Mexico', 'Philippines', 'Puerto-Rico'};
        test_clusters = {'India', 'Canada'};
        label = 'income';
        X0
        Y0
        X1
        Y1
    end

    methods
        function self = samplerReal()
            T = readtable(self.dataset, 'Delimiter',',');
            clusters = table2cell(T(:, self.clustername));
            self.X0 = T(ismember(clusters, self.train_clusters), self.features);
            self.Y0 = T(ismember(clusters, self.train_clusters), self.label);
            self.X1 = T(ismember(clusters, self.test_clusters), self.features);
            self.Y1 = T(ismember(clusters, self.test_clusters), self.label);
        end

        function [x, y] = sample(self, n0, n)
            n1 = n - n0;
            ind0 = randsample(height(self.Y0), n0);
            ind1 = randsample(height(self.Y1), n1);
            x = vertcat(self.X0(ind0, :), self.X1(ind1, :));
            y = table2cell(vertcat(self.Y0(ind0, :), self.Y1(ind1, :)));
        end
        
        function [xa, ya, xb, yb] = sample_dual(self, n0a, na, n0b, nb)
            % Samples such that no overlap between set a and b
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

