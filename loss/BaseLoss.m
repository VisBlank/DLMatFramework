classdef (Abstract) BaseLoss < handle
    %BASELOSS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = 'protected')        
    end
    
    methods (Abstract, Access = 'public')
       [loss, gradients] = GetLossAndGradients(obj, scores, targets); 
    end
    
end

