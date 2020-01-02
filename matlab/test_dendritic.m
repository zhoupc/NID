
%% 
file = '~/Downloads/Yr_d1_484_d2_477_d3_1_order_C_frames_3640_.mmap'; 
data = memmapfile(file,'Format',{'single',[3640 484, 477],'Yd'}) ; 
Yr = permute(data.Data.Yd, [3,2, 1]); 

%% 
file = '~/Downloads/Yd_pri.mmap'; 
data = memmapfile(file,'Format',{'single',[477 484 3640],'Yd'}) ; 
Y = data.Data.Yd; 

%% 
[d1, d2, T] = size(Y); 
sd = NID();
sd.change_input_size([d1, d2, 1]); 
%% 
img = neuron.reshape(img, 2); 
v_min = min(img(:));
v_max = min(abs(-10*v_min), max(img(:)));

[img_denoise, res] = sd.denoise(img, 3);
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