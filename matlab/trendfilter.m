function [ x ] = trendfilter( A, b, order, lambda )
%TRENDFILTER Fit trend filter to data, with monotonic and positive
%constraints
%   Inputs:
%       Observations: n x m
%       b: Labels (n x 1)
%       order: Regularize this order derivative
%       lambda: Regularization constant >= 0
%
%   Outputs:
%       x - Solution (m x 1)
    
    m = size(A, 2);
    pos_mono = eye(m) - diag(ones(m-1,1), 1);
    if order == 2
        trend = eye(m) - 2*diag(ones(m-1,1), 1) + diag(ones(m-2,1), 2);
    elseif order == 3
        trend = - eye(m) + 3*diag(ones(m-1,1), 1) - 3*diag(ones(m-2,1), 2) + diag(ones(m-3,1), 3);
    else
        error('Invalid input')
    end
    cvx_begin
        variable x(m)
        minimize( norm(A*x - b, 2) + lambda * norm(trend*x, 2) )
        subject to
            pos_mono*x >= 0;
    cvx_end

end

