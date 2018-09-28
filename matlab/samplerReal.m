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
            elseif strcmp(dataset, 'dota')
                self.dataset = 'data/dota/dota2Train.csv';
                self.features = {'type','hero0', 'hero1', 'hero2', 'hero3', 'hero4', 'hero5', 'hero6', 'hero7', 'hero8', 'hero9', 'hero10', 'hero11', 'hero12', 'hero13', 'hero14', 'hero15', 'hero16', 'hero17', 'hero18', 'hero19', 'hero20', 'hero21', 'hero22', 'hero23', 'hero24', 'hero25', 'hero26', 'hero27', 'hero28', 'hero29', 'hero30', 'hero31', 'hero32', 'hero33', 'hero34', 'hero35', 'hero36', 'hero37', 'hero38', 'hero39', 'hero40', 'hero41', 'hero42', 'hero43', 'hero44', 'hero45', 'hero46', 'hero47', 'hero48', 'hero49', 'hero50', 'hero51', 'hero52', 'hero53', 'hero54', 'hero55', 'hero56', 'hero57', 'hero58', 'hero59', 'hero60', 'hero61', 'hero62', 'hero63', 'hero64', 'hero65', 'hero66', 'hero67', 'hero68', 'hero69', 'hero70', 'hero71', 'hero72', 'hero73', 'hero74', 'hero75', 'hero76', 'hero77', 'hero78', 'hero79', 'hero80', 'hero81', 'hero82', 'hero83', 'hero84', 'hero85', 'hero86', 'hero87', 'hero88', 'hero89', 'hero90', 'hero91', 'hero92', 'hero93', 'hero94', 'hero95', 'hero96', 'hero97', 'hero98', 'hero99', 'hero100', 'hero101', 'hero102', 'hero103', 'hero104', 'hero105', 'hero106', 'hero107', 'hero108', 'hero109', 'hero110', 'hero111', 'hero112'};
                %self.clustername = {'cluster'};
                %self.train_clusters = [111, 112, 121, 122, 123, 124, 131, 132, 133, 134, 135, 136, 137, 138, 144, 145, 151, 152, 153, 154, 155, 156, 161, 171, 181, 182, 183, 184, 185, 186, 187, 188, 191, 192, 204, 211, 212, 213];
                %self.test_clusters = [223, 224, 225, 227, 231, 232, 241, 251, 261];
                self.clustername = {'mode'};
                self.train_clusters = [1, 2, 3, 4, 5, 6, 7, 8];
                self.test_clusters = [9];
                self.label = 'win';
                opts = detectImportOptions(self.dataset);
                opts = setvartype(opts, {'win'}, 'char');
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

