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
        m_mu = 0.9;
        m_print_every = 1000;
        m_verbose = true;
        m_data = [];
        m_loss_history = [];
        m_acc_history = [];
        m_l2_reg = 0;
        m_currEpoch = 0;
        m_bestAccuracy = 0;
        % Hold velocity state for each parameter on the model (Key will be
        % the layer name)
        m_statesWeightOptim = containers.Map('KeyType','char','ValueType','any');
        m_statesBiasOptim = containers.Map('KeyType','char','ValueType','any');
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
                    [next_w, newStateW] = obj.m_optimizer.Optimize(weight,grad.weights(keys{idx}), obj.m_statesWeightOptim(keys{idx}));
                    [next_b, newStateB] = obj.m_optimizer.Optimize(bias,grad.bias(keys{idx}), obj.m_statesBiasOptim(keys{idx}));
                    
                    % Store state info
                    obj.m_statesWeightOptim(keys{idx}) = newStateW;
                    obj.m_statesBiasOptim(keys{idx}) = newStateB;
                    
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
        
        function accHistory = GetAccuracyHistory(obj)
            accHistory = obj.m_acc_history;
        end
        
        function SetBatchSize(obj,batchSize)
            obj.m_batch_size = batchSize;
        end
        function SetEpochs(obj,epochs)
            obj.m_num_epochs = epochs;
        end
        
        function PrintEvery(obj,numIter)
            obj.m_print_every = numIter;
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
                case 'adam'
                    obj.m_optimizer = Adam(config);
                otherwise
                    fprintf('Optimizer not implemented\n');
            end
            
            % Get learning rate decay parameter (Step decay)
            if isKey(config,'lr_decay')
                obj.m_lr_decay = config('lr_decay');
            else
                obj.m_lr_decay = 1.0;
            end
            
            % Get mu parameter (Damping effect on SGD with momentum)
            if isKey(config, 'mu')
                obj.m_mu = config('mu');
            else
                obj.m_mu = 0.9;
            end
            
            % Both solver and model needs reguarization information
            obj.m_l2_reg = config('L2_reg');
            obj.m_model.L2Regularization(obj.m_l2_reg);
            
            % On solvers like SGD with momentum, ADAM, RMSProp we need to
            % store a state of the previous iteration (ex: Velocity) so we
            % need to create those before usage.
            weightsMap = obj.m_model.getWeights();
            keys = weightsMap.keys;
            biasMap = obj.m_model.getBias();
            numParameters = numel(keys);
            for idx = 1:numParameters
                weight = weightsMap(keys{idx});
                if ~isempty(weight)
                    bias = biasMap(keys{idx});
                    switch optimizerType                        
                        case 'sgd_momentum'
                            % Initialize states for SGD with momentum
                            bias_vel = zeros(size(bias), 'like',bias);
                            weight_vel = zeros(size(weight), 'like',weight);
                            obj.m_statesWeightOptim(keys{idx}) = struct('velocity',weight_vel);
                            obj.m_statesBiasOptim(keys{idx}) = struct('velocity',bias_vel);
                        case 'adam'
                            % Initialize states for Adam
                            bias_m = zeros(size(bias), 'like',bias);
                            bias_v = zeros(size(bias), 'like',bias);
                            bias_t = 0;
                            weight_t = 0;
                            weight_m = zeros(size(weight), 'like',weight);                            
                            weight_v = zeros(size(weight), 'like',weight);                            
                            obj.m_statesWeightOptim(keys{idx}) = struct('m',weight_m,'v',weight_v,'t',weight_t);
                            obj.m_statesBiasOptim(keys{idx}) = struct('m',bias_m,'v',bias_v,'t',bias_t);
                        otherwise
                            % All other stateless optimizers
                            obj.m_statesWeightOptim(keys{idx}) = [];
                            obj.m_statesBiasOptim(keys{idx}) = [];
                    end                    
                end
            end
        end
        
        function Train(obj)
            % Indicate to model that training phase started
            obj.m_model.IsTraining(true);
            
            num_train = obj.m_data.GetTrainSize();
            iterations_per_epoch = ceil(max(num_train / obj.m_batch_size, 1));
            num_iterations = ceil(obj.m_num_epochs * iterations_per_epoch);
            fprintf('Iterations/epoch: %d\n',round(iterations_per_epoch));
            
            % Indicate the dataset class that we want to auto-shuffle every
            % iterations_per_epoch iterations
            obj.m_data.shuffleEveryNIterations(iterations_per_epoch);
            
            for t=1:num_iterations
                tic;
                obj.Step();
                elapsedTime = toc;
                % Print something once and while
                if (obj.m_verbose) && (mod(t,obj.m_print_every) == 0)
                    fprintf ('(Iteration %d / %d) loss: %d stepTime: %.3f\n',(t), num_iterations, obj.m_loss_history(end), elapsedTime );
                end
                
                % At the end of the epoch increment counter and decay
                % learning rate
                epoch_end = mod((t + 1), iterations_per_epoch) == 0;
                if epoch_end
                    fprintf('Finished epoch %d/%d\n',obj.m_currEpoch+1,obj.m_num_epochs);
                    obj.m_currEpoch = obj.m_currEpoch+1;
                    % Do Step weight decay
                    currLearningRate = obj.m_optimizer.GetLearningRate();
                    currLearningRate = currLearningRate*obj.m_lr_decay;
                    obj.m_optimizer.SetLearningRate(currLearningRate);
                    
                    % Do prediction on test set if available
                    if obj.m_data.HasValidation()
                        % Get whole validation set
                        batchValidation = obj.m_data.GetValidationBatch(-1);
                        scores = obj.m_model.Predict(batchValidation.X);
                        [~, idxScoresMax] = max(scores,[],2);
                        [~, idxCorrect] = max(batchValidation.Y,[],2);
                        accuracy = mean(idxScoresMax == idxCorrect);
                        fprintf('Current Accuracy %3.3d\n',accuracy);
                        obj.m_acc_history(end+1) = accuracy;
                        if (obj.m_bestAccuracy < accuracy)
                            obj.m_bestAccuracy = accuracy;
                            currModel = obj.m_model;
                            %save('best_model.mat', 'currModel','-v7.3');
                        end
                    end
                end
            end
            
            % Indicate to model that training phase is over
            obj.m_model.IsTraining(false);
        end
    end
end

