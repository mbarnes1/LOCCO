% Where does optimization take a long (intractable) time?
clear, clc, close all

rows = [10, 100, 1000];
columns = [10, 100, 1000, 10000, 50000];
lambda = 0.1;
subsample = 1;

times = NaN(length(rows), length(columns));
statuses = cell(length(rows), length(columns));
cvx_optbnds = cell(length(rows), length(columns));
cvx_slvtol = cell(length(rows), length(columns));

for i = 1:length(rows)
    n_rows = rows(i);
    p = linspace(0, 1, n_rows);
    b = reshape(linspace(1, 0.5, n_rows), [n_rows, 1]);
    
    for j = 1:length(columns)
        n_columns = columns(j);
        [n_rows, n_columns]
        
        % Construct A
        A = NaN(n_rows, n_columns);
        for idx = 1:n_rows
            p_i = p(idx);
            D = makedist('Binomial','N',n_columns-1,'p',p_i);
            A(idx,:) = D.pdf(0:(n_columns-1));
        end
        
        % Time solution
        tic;
        %f = @() trendfilter(A, b, 4, lambda, true, subsample);
        %t_t4mono = timeit(f);
        [~, ~, solution_status] = trendfilter(A, b, 4, lambda, true, subsample);
        t_t4mono = toc;
        times(i, j) = t_t4mono;
        statuses{i, j} = solution_status{1};
        cvx_optbnds{i, j} = solution_status{2};
        cvx_slvtol{i, j} = solution_status{3};
    end
end