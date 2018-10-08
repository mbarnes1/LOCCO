clear, clc, close all

% Single trial
% load('bootstraps/synthetic/cvx_2018-10-03_08-21-47_595ce0.mat');
% 
% %% Select data to plot
% i_rows = [2];
% i_columns = [1, 2, 3, 4, 6];
% rows = rows(i_rows);
% columns = columns(i_columns);
% sketch_cvx_optbnds = sketch_cvx_optbnds(i_rows, i_columns);
% sketch_cvx_slvtol = sketch_cvx_slvtol(i_rows, i_columns);
% sketch_cvx_statuses = sketch_cvx_statuses(i_rows, i_columns);
% sketch_times = sketch_times(i_rows, i_columns);
% 
% t4mono_cvx_optbnds = t4mono_cvx_optbnds(i_rows, i_columns);
% t4mono_cvx_slvtol = t4mono_cvx_slvtol(i_rows, i_columns);
% t4mono_cvx_statuses = t4mono_cvx_statuses(i_rows, i_columns);
% t4mono_times = t4mono_times(i_rows, i_columns);
% 
% basis_times = basis_times(i_rows, i_columns);

% Multiple trials
files = {'cvx_2018-10-03_20-26-34_8f36ed.mat', ...
    'cvx_2018-10-03_20-28-11_8f36ed.mat', ...
    'cvx_2018-10-03_20-30-52_8f36ed.mat', ...
    'cvx_2018-10-03_20-31-10_8f36ed.mat', ...
    'cvx_2018-10-04_09-07-31_8f36ed.mat', ...
    'cvx_2018-10-04_09-11-44_8f36ed.mat', ...
    'cvx_2018-10-04_10-11-09_8f36ed.mat', ...
    'cvx_2018-10-04_10-16-02_8f36ed.mat', ...
    'cvx_2018-10-04_12-19-23_8f36ed.mat', ...
    'cvx_2018-10-04_12-24-36_8f36ed.mat', ...
    'cvx_2018-10-04_14-03-33_8f36ed.mat', ...
    'cvx_2018-10-04_14-07-49_8f36ed.mat'};
sketch_times = cell(length(files), 1);
t4mono_times = cell(length(files), 1);
basis_times = cell(length(files), 1);
sketch_cvx_statuses = cell(length(files), 1);
t4mono_cvx_statuses = cell(length(files), 1);

for i = 1:length(files)
    s = load(['bootstraps/synthetic/', files{i}]);
    sketch_times{i} = s.sketch_times;
    t4mono_times{i} = s.t4mono_times;
    basis_times{i} = s.basis_times;
    sketch_cvx_statuses{i} = s.sketch_cvx_statuses;
    t4mono_cvx_statuses{i} = s.t4mono_cvx_statuses;
end
sketch_times = cell2mat(sketch_times);  % length(files) x n_dataset_sizes
t4mono_times = cell2mat(t4mono_times);
basis_times = cell2mat(basis_times);
sketch_cvx_statuses = vertcat( sketch_cvx_statuses{:} );  % length(files) x n_dataset_sizes
t4mono_cvx_statuses= vertcat( t4mono_cvx_statuses{:} );

%% Solution time plot
[cmap, ~, ~] = brewermap(3, 'Set2');
linewidth = 2;
markersize = 10;

f = figure;
hold on

columns = s.columns;
linestyle = '-';
i = 1;
p1 = shadedErrorBar(columns, t4mono_times, {@mean,@std}, 'lineprops', {'Color', cmap(1,:), 'LineStyle', linestyle, 'LineWidth', linewidth, 'Marker', 'd'});
p2 = shadedErrorBar(columns, basis_times, {@mean,@std}, 'lineprops', {'Color', cmap(2,:), 'LineStyle', linestyle, 'LineWidth', linewidth, 'Marker', 'd'});
p3 = shadedErrorBar(columns, sketch_times, {@mean,@std}, 'lineprops', {'Color', cmap(3,:), 'LineStyle', linestyle, 'LineWidth', linewidth, 'Marker', 'd'});
t4mono_cvx_statuses = mean(strcmp('Solved', t4mono_cvx_statuses), 1);
sketch_cvx_statuses = mean(strcmp('Solved', sketch_cvx_statuses), 1);

for j = 1:length(columns)
    if t4mono_cvx_statuses(j) < 0.5
        p4 = plot(columns(j), t4mono_times(i, j), 'k*', 'MarkerSize', markersize, 'LineWidth', linewidth);
    else
        p5 = plot(columns(j), t4mono_times(i, j), 'd', 'Color', cmap(1,:), 'LineWidth', linewidth);
    end

    if sketch_cvx_statuses(j) < 0.5
        p4 = plot(columns(j), sketch_times(i, j), 'k*', 'MarkerSize', markersize, 'LineWidth', linewidth);
    else
        p5 = plot(columns(j), sketch_times(i, j), 'd', 'Color', cmap(3,:), 'LineWidth', linewidth);
    end

end

xlabel('Training samples n')
ylabel('Solution time (s)')
set(gca, 'XScale', 'log', 'YScale', 'log');
legend([p1.mainLine, p2.mainLine, p3.mainLine, p4], 'T4+mono', 'Basis', 'Sketch', 'Failed', 'Location', 'northwest')
set(f, 'units', 'inches', 'pos', [0 0 6 4.5])