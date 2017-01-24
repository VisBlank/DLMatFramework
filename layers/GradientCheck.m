classdef GradientCheck < handle
    % Evaluate gradient numerically(slow), this class is used to debug
    % backpropagation.
    % References:
    % https://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking
    % https://uk.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html
    % http://cs231n.github.io/optimization-1/#numerical
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment1/cs231n/gradient_check.py
    % Ex:
    % sigmoid_forward = @(x) (1./(1+exp(-x)));
    % sigmoid_backward = @(x) 0.1e1 ./ (0.1e1 + exp(-x)).^2 .*exp(-x);
    % sigmoid_backward(0)
    % GradientCheck.Eval(sigmoid_forward,0,1)
    % sigmoid_backward([0 1 2])
    % GradientCheck.Eval(sigmoid_forward,[0 1 2],1)
    
    properties
    end
    
    methods (Access = 'public', Static)
        function grad = Eval(f,x,dout)
            % Check if is a scalar
            if prod(size(x)) == 1
                grad = GradientCheck.Eval_scalar_input(f, x, dout);
            else
                grad = GradientCheck.Eval_vector_input(f, x, dout);
            end
        end
        
        function grad = Eval_vector_input(f,x,dout)
            % f will be a lambda of a function with a single parameter
            % x point(on any dimension) where to evalualte the gradient            
            h = 0.00001;
            grad = zeros(size(x));
            
            % Iterate on all dimensions of the vector x
            for i=1:numel(x)   
                old_value = x(i);
                x(i) = old_value + h;
                
                % Evaluate f(x+h)
                f_x_plus_h = f(x);
                
                x(i) = old_value - h;
                
                % Evaluate f(x+h)
                f_x_minus_h = f(x);
                
                % Restore the previous value
                x(i) = old_value;
                
                % Now compute the partial derivative
                grad(i) = sum((f_x_plus_h-f_x_minus_h) .* dout)./(2*h);
                
            end
        end
        % This function can be accelerated if we vectorize it
        function grad = Eval_scalar_input(f, x, dout)            
            % f will be a lambda of a function with a single parameter
            % x point where to evalualte the gradient            
            h = 0.00001;
            % Remember that the gradient of f w.r.t of x will have the same
            % dimensions of x
            grad = zeros(size(x));
                        
            % Iterate on all dimensions of the vector x
            for i=1:numel(x)                
                old_value = x(i);
                x(i) = old_value + h;
                
                % Evaluate f(x+h)
                f_x_plus_h = f(x);
                
                x(i) = old_value - h;
                
                % Evaluate f(x+h)
                f_x_minus_h = f(x);
                
                % Restore the previous value
                x(i) = old_value;
                
                % Now compute the partial derivative
                grad(i) = ((f_x_plus_h-f_x_minus_h).*dout)./(2*h);
                %grad(i) = grad(i) * dout;
            end
        end
    end
    
end

