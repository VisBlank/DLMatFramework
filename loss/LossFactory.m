classdef (Sealed) LossFactory < handle
    %LOSSFACTORY Summary of this class goes here
    % References: https://uk.mathworks.com/help/matlab/matlab_oop/controlling-the-number-of-instances.html
    
    properties
    end
    
    methods (Static)
        % Singleton pattern for loss function
        function lossInstance = GetLoss(lossType)
            persistent localObj
            % After the first loss is created it will not create anymore
            if isempty(localObj) || ~isvalid(localObj)                
                switch lossType
                    case 'cross_entropy'
                        localObj = CrossEntropy();
                    case 'mean_squared_error'
                        localObj = MeanSquaredError();
                    case 'multi_class_cross_entropy'
                        localObj = MultiClassCrossEntropy();
                    otherwise
                        localObj =  [];
                end
            end
            lossInstance = localObj;
        end
    end
    
end

