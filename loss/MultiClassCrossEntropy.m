classdef MultiClassCrossEntropy < BaseLoss
    % Multi-class cross entropy
    % References:
    % https://www.ics.uci.edu/~pjsadows/notes.pdf
    
    properties
    end
    
    methods (Access = 'public')
        function [loss, gradients] = GetLossAndGradients(obj, prob, targets)
            N = size(prob,1);
            % Considering that the scores are already converted to
            % probabilities.
            % Get the indexes of the correct classes
            [~, idxCorrect] = max(targets,[],2);
            
            % Get the probabilities of the correct classes
            probCorrect = diag(prob(:,idxCorrect));

            loss = -sum(log(probCorrect))/N;            
            
            % Get gradient of loss w.r.t to the correct score
            gradients = prob;
            gradients_correct = probCorrect - 1;
            % Put the calculated correction back on 
            gradients(sub2ind(size(gradients),[1:length(idxCorrect)]',idxCorrect)) = gradients_correct;            
            gradients = gradients / N;
        end
    end    
end

