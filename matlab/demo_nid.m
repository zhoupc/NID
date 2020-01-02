sd = NID();

load(fullfile('../', 'datasets', 'testdata', 'baylor_data.mat'));

%%
A_raw = A_corr;
[d1, d2, d3, K] = size(A_raw);
A_raw = reshape(A_raw, d1, d2, []);
%%
for example_id = 1:size(A_raw, 3)
    %%
    img = A_raw(:, :, example_id);
    v_min = min(img(:));
    v_max = min(-10*v_min, max(img(:)));
    
    img = img + randn(size(img))*abs(v_min)/3;
    [img_denoise, res] = sd.denoise_test(img, 3);
    n = 5;
    [d1, d2] = size(img);
    figure('papersize', [d2, d1*2]);
    
    
    subplot(321);
    imagesc(img, [v_min, v_max]);
    colorbar;
    title('raw image');
    axis equal off tight;
    
    subplot(323);
    imagesc(img_denoise, [0, v_max]);
    %     imagesc(res{1}, [v_min, -v_min]);
    colorbar;
    title('two-step');
    axis equal off tight;
    
    subplot(325) ;
    imagesc(img-img_denoise, [v_min, -v_min]);
    %     imagesc(res{2}, [v_min, -v_min]);
    colorbar;
    axis equal off tight;
    title('residual');
    
subplot(3,2,[2,4,6]); 
plot(img(:), img_denoise(:), '.');
    hold on;
    plot(img(:), img(:), 'r');
    axis tight equal;
    xlabel('raw');
    ylabel('denoised');
    set(gca, 'yaxislocation', 'right');
    
    pause;
    %%
    %     saveas(gcf, sprintf('test_baylor/cell_%d.pdf', example_id));
    %     pause(0.5);
    close all;
end