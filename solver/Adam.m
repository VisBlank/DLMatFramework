classdef Adam < Optimizer
    % Adam: First-order gradient-based optimization of stochastic objective
    % functions, based on adaptive estimates of lower-order moment
    % https://arxiv.org/pdf/1412.6980.pdf
    % http://cs231n.github.io/neural-networks-3/#ada
    % https://github.com/leonardoaraujosantos/DLMatFramework/issues/13
    % adam = Adam(containers.Map({'learning_rate','beta1','beta2'}, {0.1,0.9,0.999}));
    
    properties(Access = protected)
        m_config = containers.Map();
        m_base_lr = 1e-3;
        m_beta1 = 0.9;  
        m_beta2 = 0.999;        
        % Avoid numerical instability on sqrt calculation
        m_epsilon = 1e-8;
    end
    
    methods(Access = public)
        % Constructor
        function obj = Adam(config)
            obj.m_config = config;
            obj.m_base_lr = obj.m_config('learning_rate');
            % Check if beta1 is part if map keys
            if ismember('beta1', obj.m_config.keys)
                obj.m_beta1 = obj.m_config('beta1');
            end
            % Check if beta1 is part if map keys
            if ismember('beta2', obj.m_config.keys)
                obj.m_beta2 = obj.m_config('beta2');
            end
        end
        
        function [weights, newState] = Optimize(obj, w, dw, state)
            weights = 0;
            newState = 0;
            
        end
    end
    
end

