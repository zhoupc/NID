classdef NID < handle 
    
    properties 
        net = [];
        nid_folder = '';    % the folder 
        net_name = 'net_17'; 
        iter = 2; 
    end 
    
    
    methods 
        function obj = NID()
            nid_folder = fileparts(mfilename('fullpath'));
            net_file = fullfile(nid_folder, 'logs', [obj.net_name, '.onnx']); 
            obj.net = importONNXLayers(net_file, 'OutputLayerType', 'regression');
        end 
        
        %% denoise 
        function img = denoise(obj, img)
            for m=1:obj.iter
                % denoise 
            end 
        end 
        
    end 
    
end 