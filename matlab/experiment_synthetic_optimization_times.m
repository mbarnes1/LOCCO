% Where does optimization take a long (intractable) time?
clear, clc, close all

git = getGitInfo();
git = git.hash(1:6);

rows = [10, 100, 1000];
columns = [10, 100, 1000, 10000, 50000, 100000];
lambda = 0.1;
subsample = 1;
sketch_factor = 10;

t4mono_times = NaN(length(rows), length(columns));
t4mono_cvx_statuses = cell(length(rows), length(columns));
t4mono_cvx_optbnds = cell(length(rows), length(columns));
t4mono_cvx_slvtol = cell(length(rows), length(columns));

sketch_times = NaN(length(rows), length(columns));
sketch_cvx_statuses = cell(length(rows), length(columns));
sketch_cvx_optbnds = cell(length(rows), length(columns));
sketch_cvx_slvtol = cell(length(rows), length(columns));

basis_times = NaN(length(rows), length(columns));

polyorder = 6;

for i = 1:length(rows)
    n_rows = rows(i);
    p = linspace(0, 1, n_rows);
    b = reshape(linspace(1, 0.5, n_rows), [n_rows, 1]);
    
    for j = 1:length(columns)
        n_columns = columns(j);
        [n_rows, n_columns]
        sketch = floor(n_columns / sketch_factor);
        
        % Construct A
        A = NaN(n_rows, n_columns);
        for idx = 1:n_rows
            p_i = p(idx);
            D = makedist('Binomial','N',n_columns-1,'p',p_i);
            A(idx,:) = D.pdf(0:(n_columns-1));
        end
        
        % T4 + mono
        tic;
        %f = @() trendfilter(A, b, 4, lambda, true, subsample);
        %t_t4mono = timeit(f);
        [~, ~, solution_state] = trendfilter(A, b, 4, lambda, true, subsample);
        t_t4mono = toc;
        t4mono_times(i, j) = t_t4mono;
        t4mono_cvx_statuses{i, j} = solution_state{1};
        t4mono_cvx_optbnds{i, j} = solution_state{2};
        t4mono_cvx_slvtol{i, j} = solution_state{3};
        
        % Basis
        tic;
        [~] = polyfilter(A, b, polyorder);
        t_basis = toc;
        basis_times(i, j) = t_basis;
        
        % Sketch
        tic;
        [~,~, solution_state] = filter_sketch(A, b, sketch, 'block', lambda, true, subsample);
        t_sketch = toc;
        sketch_times(i, j) = t_sketch;
        sketch_cvx_statuses{i, j} = solution_state{1};
        sketch_cvx_optbnds{i, j} = solution_state{2};
        sketch_cvx_slvtol{i, j} = solution_state{3};
    end
end

filename = ['cvx_', datestr(now, 'yyyy-mm-dd_hh-MM-ss'), '_', git, '.mat'];
save(filename, 'sketch_times', 'sketch_cvx_statuses', 'sketch_cvx_optbnds', 'sketch_cvx_slvtol', ...
    't4mono_times', 't4mono_cvx_statuses', 't4mono_cvx_optbnds', 't4mono_cvx_slvtol', ...
    'basis_times', 'rows', 'columns', 'git')