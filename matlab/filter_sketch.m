function [ x, res, vargout ] = filter_sketch( A, b, s, method, lambda, mono, subsample )
%FILTER_SKETCH Solve system using matrix sketching.
%   Inputs:
%       A: n x m
%       b: n x 1 observed bootstrap errors
%       s: Int less than or equal to n, sketch matrix to size s x m. 
%       lambda: Regularization constant >= 0
%       mono: True/False, whether to enforce monotonic constraint
%   Outputs:
%       x - Solution (m x 1) to sketched linear system.
switch method
    case 'medoid'
        [S, idx] = sketch_medoid(A, s);
    case 'block'
        [S, idx] = sketch_block(A, s);
    case 'mean'
        [S, idx] = sketch_mean(A, s);
    otherwise
        error('Invalid sketching method.')
end
if or(lambda > 0, mono)
    [x, res, vargout] = trendfilter(S, b, 2, lambda, mono, subsample);
else
    x = S \ b;
    res = NaN;
end
x = x(idx);

end

function [ S, idx ] = sketch_medoid( A, s )
%SKETCH_MEDOID Sketched matrix A using k-medoids.
%   Inputs:
%       A: n x m
%       s: Int less than or equal to n, sketch matrix to size n x s. 
%   Outputs:
%       S   - n x s sketched version of S
%       idx - m x 1 vector of mapping from original column indices to 
%             sketched column indices
    
    [idx, centers] = kmedoids(transpose(A(:, 2:end)), s-1);
    S = [A(:,1), transpose(centers)];
    idx = [1; idx+1];
    cluster_sizes = count_cluster_sizes(idx);
    S = S .* cluster_sizes;
end


function [ S, idx ] = sketch_block( A, s )
%SKETCH_BLOCK Sketched matrix A using basic blocking.
%   Inputs:
%       A: n x m
%       s: Int less than or equal to n, sketch matrix to size n x s. 
%   Outputs:
%       S - n x s sketched version of S
%       idx - m x 1 vector of mapping from original column indices to 
%             sketched column indices

    m = size(A, 2);
    idx = round(linspace(1, s, m));  % old column --> new column (onto)
    centers = round(linspace(1, m, s));  % new column --> old column representative, the center
    S = A(:, centers);
    cluster_sizes = count_cluster_sizes(idx);
    S = S .* cluster_sizes;
end


function [S, idx ] = sketch_mean( A, s)
%SKETCH_MEAN Sketched matrix A using basic blocking.
%   Inputs:
%       A: n x m
%       s: Int less than or equal to n, sketch matrix to size n x s. 
%   Outputs:
%       S - n x s sketched version of S
%       idx - m x 1 vector of mapping from original column indices to 
%             sketched column indices

    m = size(A, 2);
    idx = round(linspace(1, s, m));  % old column --> new column (onto)
    S = zeros(size(A, 1), s);
    for old_column_idx = 1:length(idx)
        new_column_idx = idx(old_column_idx);
        S(:, new_column_idx) = S(:, new_column_idx) + A(:, old_column_idx);
    end
end


function [cluster_sizes] = count_cluster_sizes(idx)
%COUNT_CLUSTER_SIZES
%   Inputs:
%       idx: Vector mapping sample to cluster_id
%
%   Outputs:
%       cluster_sizes: Vector mapping cluster_id to counts
max_cluster_id = max(idx);
cluster_sizes = zeros(1, max_cluster_id);
for i = 1:length(idx)
    cluster_sizes(idx(i)) = cluster_sizes(idx(i)) + 1;
end

end