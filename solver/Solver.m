classdef Solver < handle
    %SOLVER Encapsulate all the logic for training, like, separating stuff
    % on batches, shuffling the dataset, and updating the weights with a
    % policy defined on the class optimizer
    % Example:
    % myModel = DeepLearningModel(layers);
    % solver = Solver(myModel, data, 'sgd',containers.Map({'learning_rate'}, {0.1}));
    % solver.train();
    % myModel.loss(X);
    % Reference:
    % https://github.com/leonardoaraujosantos/DLMatFramework/blob/master/learn/cs231n/assignment2/cs231n/solver.py
    
    properties(Access = protected)
        m_optimizer;
        m_model;
        m_num_epochs = 10;
        m_batch_size = 100;
        m_lr_decay = 1;
        m_print_every = 1000;
        m_verbose = true;
    end
    
    methods(Access = protected)
        function Step(obj)
            % Make a single gradient update. This is called by train() and should not
            % be called manually.
            
            %% Select a mini-batch
            
            %% Get model loss and gradients(dw)
            
            %% Perform a parameter update
        end
    end
    
    methods(Access = public)
        function obj = Solver(model, data, optimizerType, config)
            obj.m_model = model;
            switch optimizerType
                case 'sgd'
                    obj.m_optimizer = Sgd(config);
                case 'sgd_momentum'
                    obj.m_optimizer = SgdMomentum(config);
                otherwise
                    fprintf('Optimizer not implemented\n');
            end
        end
        
        function Train(obj)
            fprintf('Need implementation\n');
        end
        
        function [accuracy] = CheckAccuracy(obj)
            accuracy = 0;
        end
    end
    
end

