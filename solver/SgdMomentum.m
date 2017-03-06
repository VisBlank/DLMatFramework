classdef SgdMomentum < Optimizer
    % Stochastic gradient descent
    % sgdM = SgdMomentum(containers.Map({'learning_rate','momentum'}, {0.1,0.9}));
    
    properties(Access = protected)
        m_config = containers.Map();
        m_base_lr = 0;
        m_momentum = 0.9;        
    end
    
    methods(Access = public)
        % Constructor
        function obj = SgdMomentum(config)
            obj.m_config = config;
            obj.m_base_lr = obj.m_config('learning_rate');
            % Check if momentum is part if map keys
            if ismember('momentum', obj.m_config.keys)
                obj.m_momentum = obj.m_config('momentum');
            end
        end
        
        function [weights, newState] = Optimize(obj, w, dw, state)            
            next_w = w;
            % Velocity will be a moving average of the gradients.
            state.velocity = obj.m_momentum * state.velocity - obj.m_base_lr*dw;
            
            % Update weights
            next_w = next_w + state.velocity;
            weights = next_w;  
            
            % Update state
            newState = state;
        end
    end
    
end

