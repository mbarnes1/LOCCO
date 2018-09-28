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
            if strcmp(dataset, 'adult')
                self.dataset = 'data/adult/adult.csv';
                self.features = {'age', 'education_num', 'hours_per_week', 'race', 'occupation'};
                self.clustername = {'native_country'};
                self.train_clusters = {'United-States', 'El-Salvador', 'Germany', 'Mexico', 'Philippines', 'Puerto-Rico'};
                self.test_clusters = {'India', 'Canada'};
                self.label = 'income';
                opts = detectImportOptions(self.dataset);
            elseif strcmp(dataset, 'heart')
                self.dataset = 'data/heart/heart_all.csv';
                self.features = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'thal'};
                self.clustername = {'country'};
                self.train_clusters = {'cleveland', 'va', 'switzerland'};
                self.test_clusters = {'hungarian'};
                self.label = 'num';
                opts = detectImportOptions(self.dataset);
            elseif strcmp(dataset, 'parkinson')
                self.dataset = 'data/parkinson/train_data.csv';
                self.features = {'jitter_local', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_local', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'ac', 'nth', 'htn', 'median_pitch', 'mean_pitch', 'std_dev', 'min_pitch', 'max_pitch', 'pulses', 'periods', 'mean_period', 'std_dev_period', 'unvoiced', 'breaks', 'deg_breaks'}; %, 'UPDRS'};
                self.clustername = {'subject'};
                self.train_clusters = 1:40;
                self.test_clusters = 1:4:40;
                self.train_clusters(self.test_clusters) = [];
                self.label = 'class';
                opts = detectImportOptions(self.dataset);
                opts = setvartype(opts, {'class'}, 'char');
            else
                error('Invalid dataset name')
            end
            opts.Delimiter = ',';
            T = readtable(self.dataset, opts);
            clusters = table2cell(T(:, self.clustername));
            if ~ischar(clusters{1})
                clusters = cell2mat(clusters);
            end
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

