classdef (Abstract) Optimizer < handle
    % Base class for optimizers (ie: sgd, sgd_momentun, adam)
    % Referece:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/optim.py
    
    % All derived classes must have those properties defined
    properties (Abstract, Access = protected)
        m_config    % General configuration
        m_base_lr   % Learning rate used when training starts
    end
    
    % All derived classes must have those methods implemented
    methods(Abstract, Access = public)
        [weights, newState] = Optimize(obj, w, dw, state);
    end
    
    methods (Access = public)
        function SetLearningRate(obj, lrRate)
           obj.m_base_lr = lrRate;
        end
        function [lr] = GetLearningRate(obj)
           lr = obj.m_base_lr;
        end
    end
    
end

