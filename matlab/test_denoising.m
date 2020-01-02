sd = NID();

load datasets/testdata/real_2pdata.mat;

%%
for example_id = 1:size(A_raw, 3)
    %%
    img = A_raw(:, :, example_id);
    v_min = min(img(:));
    v_max = min(-10*v_min, max(img(:)));
    
    img = img + randn(size(img))*abs(v_min)/3; 
    [img_denoise, res] = sd.denoise(img, 2);
    n = 6;
    figure('papersize', [18, 2.8]);
    init_fig;
    axs = tight_subplot(1, n);
    axes(axs(1));
    imagesc(img, [v_min, v_max]);
    colorbar;
    title('raw image');
    axis equal off tight;
    
    axes(axs(2));
    imagesc(img-res{1}, [0, v_max]);
    colorbar;
    title('1st iter');
    axis equal off tight;
    
    axes(axs(3));
    imagesc(img-res{2}, [0, v_max]);
    colorbar;
    title('2nd iter');
    axis equal off tight;
    
    axes(axs(4));
    imagesc(img_denoise, [0, v_max]);
    colorbar;
    title('two-step');
    axis equal off tight;
    
    axes(axs(5));
    imagesc(img-img_denoise, [v_min, -v_min]);
    colorbar;
    axis equal off tight;
    title('residual');
    
    axs(6).Position = axs(6).Position + [0, 0.08, 0, 0]; 
    axes(axs(6));
    plot(img(:), img_denoise(:), '.');
    hold on;
    plot(img(:), img(:), 'r');
    axis tight equal;
    xlabel('raw');
    ylabel('denoised');
    set(gca, 'yaxislocation', 'right');
    %%
    saveas(gcf, sprintf('test_folder/cell_%d.pdf', example_id));
    pause(0.5); 
    close;
end