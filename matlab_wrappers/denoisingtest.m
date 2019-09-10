results = load('scan_1.mat')
results = results.results
A = results.A;
C = results.C;
AM = results.A_mask;
scan_plane = 1;
scan_number = 34;
filePath = 'data/home/app2139/NID/logs/net_17.pth';

for i = 1:387 
    img = A(:,:,scan_plane, i);
    img = imnoise(img, 'gaussian', 0, 0.01);
    %I = imshow(img, 'InitialMagnification', 700)
    J = imadjust(img);
    %imshow(J, 'InitialMagnification', 700);
    
    dims = size(img)
    vector = img(:)
    
    img2 = py.denoiseMAT.denoiseit(vector, dims(1),dims(2), py.str(filePath));
    data = double(py.array.array('d',py.numpy.nditer(img2)));
    J2 = reshape(data, dims(1), dims(2))
    
    imshow([J,J2], 'InitialMagnification', 700);
    w = waitforbuttonpress;
    close all;
    disp(i)
end