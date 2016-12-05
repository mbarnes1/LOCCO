classdef sampler < handle
    %SAMPLER Sample from some mix of the training and testing distributions
    
    properties (SetAccess = private)
        m0
        m1
        sigma0
        sigma1
        xrange0
        xrange1
    end
    
    methods
        function self = sampler(m0, m1, sigma0, sigma1, xrange0, xrange1)
            self.m0 = m0;
            self.m1 = m1;
            self.sigma0 = sigma0;
            self.sigma1= sigma1;
            self.xrange0 = xrange0;
            self.xrange1 = xrange1;
        end
        
        function [x, y] = sample(self, n0, n)
            n1 = n - n0;
            x0 = rand([n0, 1])*(self.xrange0(1) - self.xrange0(2)) + self.xrange0(2);
            x1 = rand([n1, 1])*(self.xrange1(1) - self.xrange1(2)) + self.xrange1(2);
            y0 = x0*self.m0 + normrnd(0, self.sigma0, [n0, 1]);
            y1 = x1*self.m1 + normrnd(0, self.sigma1, [n1, 1]);
            x = [x0;
                x1];
            y = [y0;
                y1];
        end
    end
    
end

