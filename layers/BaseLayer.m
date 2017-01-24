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
        index
        activationShape
        inputLayer        
    end
    
    properties (Access = 'public')
       doGradientCheck = false; 
    end
    
    methods(Abstract, Access = 'public')
        % Activations will be a tensor
        [activations] = ForwardPropagation(obj, inputs, weights, bias);
        % Gradient will be a struct with input, bias, weight
        [gradient] = BackwardPropagation(obj);  
        [gradient] = EvalBackpropNumerically(obj, dout);
    end
    
    methods(Access = 'public')
       function [activations] = getActivations(obj)
            activations = obj.activations;
       end 
       
       function [config] = getConfig(obj)
            config = obj.config;
       end 
       
       function [index] = getIndex(obj)
            index = obj.index;
       end 
       
       function [name] = getName(obj)
            name = obj.name;
       end 
       
       function [actShape] = getActivationShape(obj)
           actShape = obj.activationShape;
       end
       
       function [layer] = getInputLayer(obj)
          layer = obj.inputLayer; 
       end
       
       function EnableGradientCheck(obj, flag)
           obj.doGradientCheck = flag;
       end
              
    end
    
end