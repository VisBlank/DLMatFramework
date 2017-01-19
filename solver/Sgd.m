classdef Sgd < Optimizer
    % Stochastic gradient descent
    % sgd = Sgd(containers.Map({'learning_rate'}, {0.1}));
    
    properties(Access = protected)
        m_config = containers.Map();
        m_base_lr = 0;
    end
    
    methods(Access = public)
        % Constructor
        function obj = Sgd(config)
            obj.m_config = config;
            obj.m_base_lr = obj.m_config('learning_rate');
        end
        
        function [weights] = Optimize(obj, w, dw)
            % Gradient descent approach (Simplest optimizer)
            w = w - (obj.m_base_lr * dw);
            weights = w;
        end
    end
    
end

