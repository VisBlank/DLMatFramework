classdef LossFactory < handle
    %LOSSFACTORY Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Static)
        function lossInstance = GetLoss(lossType)
            switch lossType
                case 'cross_entropy'
                    lossInstance = CrossEntropy();
                case 'mean_squared_error'
                    lossInstance = MeanSquaredError();
                otherwise
                    lossInstance =  [];
            end
        end
    end
    
end

