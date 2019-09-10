function [denoised] = denoiseNN(img,x,y,filePath)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    dims = size(img)
    vector = img(:) %Can only pass vectors to python ...
    
    img2 = py.denoiseMAT.denoiseit(vector, dims(1),dims(2), py.str(filePath));
    data = double(py.array.array('d',py.numpy.nditer(img2)));
    denoised = reshape(data, dims(1), dims(2))
    
end

