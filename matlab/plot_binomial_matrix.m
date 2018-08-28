clear, clc, close all
output_dir = '/Users/mbarnes1/Documents/trafficjam/thesis/phd-talk/figures/';

% Params
n_levels = 20;
n_samples = 39;
p_0 = 0.1;
p_f = 1.0;
pdf_matrix = nan(n_levels, n_samples+1);
counter = 1;
for p = linspace(p_0, p_f, n_levels)
    pdf = binopdf(0:n_samples, n_samples, p);
    pdf = pdf / max(pdf);
    pdf_matrix(counter, :) = pdf;
    counter = counter + 1;
end
imwrite(size(colormap, 1)*pdf_matrix, colormap, strcat(output_dir, 'binomial_matrix.eps'))
%image(pdf_matrix, 'CDataMapping', 'scaled')
%axis off
%set(gcf, 'Position', [440   378   560   420/2])

% After sketching by factor of 2
pdf_matrix_sketched = nan(n_levels, size(pdf_matrix, 2) / 2);
for i = 1:size(pdf_matrix_sketched, 2)
    column_indices = [2*i - 1, 2*i];
    pdf_matrix_sketched(:, i) = mean(pdf_matrix(:, column_indices), 2);
end
imwrite(size(colormap, 1)*pdf_matrix_sketched, colormap, strcat(output_dir, 'binomial_matrix_sketched.eps'))
% figure
% image(pdf_matrix_sketched, 'CDataMapping', 'scaled')
% axis off
% set(gcf, 'Position', [440   378   560/2   420/2])

    
    