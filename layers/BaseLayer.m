classdef (Abstract) BaseLayer < handle
    %BASELAYER Abstract class for Layer
    % Check format on reference project:
    % http://cs.stanford.edu/people/karpathy/convnetjs/
    % https://github.com/karpathy/convnetjs 
    % https://databoys.github.io/Feedforward/
    % http://scs.ryerson.ca/~aharley/neural-networks/
    
    properties (Abstract, Access = 'protected')                
        weights
        biases
        activations
        config
        previousInput
        name
    end
    
    methods(Abstract, Access = 'public')
        [activations] = ForwardPropagation(obj, inputs, weights, bias);                                
        [gradient] = BackwardPropagation(obj);                
    end
    
    methods(Access = 'public')
       function [activations] = getActivations(obj)
            activations = obj.activations;
       end 
       
       function [config] = getConfig(obj)
            config = obj.config;
       end 
       
       function [name] = getName(obj)
            name = obj.name;
       end 
              
    end
    
end