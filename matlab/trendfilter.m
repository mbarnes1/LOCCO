function [ x, res, vargout ] = trendfilter( A, b, order, lambda, mono, subsample )
%TRENDFILTER Fit trend filter to data, with monotonic and positive
%constraints
%   Inputs:
%       Observations: n x m
%       b: Labels (n x 1)
%       order: Regularize this order derivative
%       lambda: Regularization constant >= 0
%       mono: True/False, whether to enforce monotonic constraint
%       subsample: Factor to downsample the number of corruption levels in
%       A and b
%   Outputs:
%       x - Solution (m x 1)
%       res - Residual
%       vargout - Optionally, return the cvx_status
    
    n = size(A, 1);
    n_new = floor(n / subsample);
    indices = floor(linspace(1, n, n_new));
    A = A(indices, :);
    b = b(indices, :);
    
    m = size(A, 2);
    pos_mono = eye(m) - diag(ones(m-1,1), 1);
    if order == 2
        trend = eye(m) - 2*diag(ones(m-1,1), 1) + diag(ones(m-2,1), 2);
        trend = trend(1:end-2, :);
    elseif order == 3
        trend = - eye(m) + 3*diag(ones(m-1,1), 1) - 3*diag(ones(m-2,1), 2) + diag(ones(m-3,1), 3);
        trend = trend(1:end-3, :);
    elseif order == 4
        trend = eye(m) - 4*diag(ones(m-1,1), 1) + 6*diag(ones(m-2,1), 2) -4*diag(ones(m-3,1), 3) + diag(ones(m-4,1), 4);
        trend = trend(1:end-4, :);
    else
        error('Invalid input')
    end
    if mono
        cvx_begin
            variable x(m)
            minimize( norm(A*x - b, 2) + lambda * norm(trend*x, 2) )
            subject to
                pos_mono*x >= 0;
        cvx_end
    else
        cvx_begin
            variable x(m)
            minimize( norm(A*x - b, 2) + lambda * norm(trend*x, 2) )
        cvx_end
    end
    res = norm(A*x - b, 2);
    if nargout > 2
        vargout{1} = cvx_status;
        vargout{2} = cvx_optbnd;
        vargout{3} = cvx_slvtol;
    end
end