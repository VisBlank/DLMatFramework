classdef (Sealed) LossFactory < handle
    %LOSSFACTORY Summary of this class goes here
    % References: https://uk.mathworks.com/help/matlab/matlab_oop/controlling-the-number-of-instances.html
    
    properties
    end
    
    methods (Static)
        % Singleton pattern for loss function
        function lossInstance = GetLoss(lossType)
            persistent localObj
            if isempty(localObj) || ~isvalid(localObj)                
                switch lossType
                    case 'cross_entropy'
                        localObj = CrossEntropy();
                    case 'mean_squared_error'
                        localObj = MeanSquaredError();
                    otherwise
                        localObj =  [];
                end
            end
            lossInstance = localObj;
        end
    end
    
end

