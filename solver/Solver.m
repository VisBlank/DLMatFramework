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
        m_data = [];
        m_loss_history = [];
        m_l2_reg = 0;
    end
    
    methods(Access = protected)
        function Step(obj)
            % Make a single gradient update. This is called by train() and should not
            % be called manually.
            
            %% Select a mini-batch
            batch = obj.m_data.GetBatch(obj.m_batch_size);
            X_batch = batch.X;
            Y_batch = batch.Y;
            
            %% Get model loss and gradients(dw)
            [loss, grad] = obj.m_model.Loss(X_batch, Y_batch);
            if isa(loss,'gpuArray')
                obj.m_loss_history(end+1) = gather(loss);
            else
                obj.m_loss_history(end+1) = loss;
            end            
            
            %% Perform a parameter update
            weightsMap = obj.m_model.getWeights();
            keys = weightsMap.keys;
            biasMap = obj.m_model.getBias();            
            numParameters = numel(keys);
            for idx = 1:numParameters
                weight = weightsMap(keys{idx});
                if ~isempty(weight)
                    bias = biasMap(keys{idx});                    
                    
                    % Add regularization gradient contribution
                    grad.weights(keys{idx}) = grad.weights(keys{idx}) + (obj.m_l2_reg * weight);
                    
                    % Use optimizer to calculate new weights
                    next_w = obj.m_optimizer.Optimize(weight,grad.weights(keys{idx}));
                    next_b = obj.m_optimizer.Optimize(bias,grad.bias(keys{idx}));
                    
                    % Update weights on model
                    weightsMap(keys{idx}) = next_w;
                    biasMap(keys{idx}) = next_b;
                end
            end
        end
    end
    
    methods(Access = public)
        function lossHistory = GetLossHistory(obj)
            lossHistory = obj.m_loss_history;
        end
        function SetBatchSize(obj,batchSize)
            obj.m_batch_size = batchSize;
        end
        function SetEpochs(obj,epochs)
            obj.m_num_epochs = epochs;
        end
        
        function obj = Solver(model, data, optimizerType, config)
            obj.m_model = model;
            % Get reference to your training data
            obj.m_data = data;
            switch optimizerType
                case 'sgd'
                    obj.m_optimizer = Sgd(config);
                case 'sgd_momentum'
                    obj.m_optimizer = SgdMomentum(config);
                otherwise
                    fprintf('Optimizer not implemented\n');
            end
            
            % Both solver and model needs reguarization information
            obj.m_l2_reg = config('L2_reg');
            obj.m_model.L2Regularization(obj.m_l2_reg);
        end
        
        function Train(obj)
            num_train = obj.m_data.GetTrainSize();
            iterations_per_epoch = max(num_train / obj.m_batch_size, 1);
            num_iterations = obj.m_num_epochs * iterations_per_epoch;
            
            for t=1:num_iterations
                obj.Step();
                if (obj.m_verbose) && (mod(t,obj.m_print_every) == 0)
                    fprintf ('(Iteration %d / %d) loss: %d\n',(t + 1), num_iterations, obj.m_loss_history(end) );
                end
            end
        end
        
        % Check accuracy of model with some given dataset
        function [accuracy] = CheckAccuracy(obj, X, Y, num_samples, batchSize)
            % Get 4-d tensor batch size
            N = size(X,4);
            
            % Subsample the data (also shuffle)
            if (~isempty(num_samples)) && (num_samples>N)
                mask = randperm(N);
                selIndex = mask(1:num_samples);
                X = X(:,:,:,selIndex);
                Y = Y(selIndex,:);
                N = num_samples;
            end
            
            % Compute predictions in batches
            num_batches = N / batchSize;
            if mod(N,batchSize) ~= 0
                num_batches = num_batches+1;
            end
            y_pred = [];
            for i=1:num_batches
                start_idx = i * batchSize;
                end_idx = (i+1) * batchSize;
                scores = obj.m_model.Predict(X(:,:,:,start_idx:end_idx));
                [~,y_pred(end+1)] = max(scores);
            end
            
            accuracy = mean(y_pred == Y);
        end
        
    end
    
end

