% clearvars;
% tic
%fprintf('Started Script');

%fprintf('\nBefor Calling Preprocess from Script.m');
[train_data, train_label,train_label_mat,validation_data, ...
    validation_label, test_data,test_label] = preprocess();

%fprintf('\nAfter Calling Preprocess from Script.m');

save('dataset.mat', 'train_data', 'train_label','train_label_mat', 'validation_data', ...
                    'validation_label','test_data', 'test_label');
load('dataset.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**************Neural Network********************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Train Neural Network

% set the number of nodes in input unit (not including bias unit)
n_input = size(train_data, 2); 

% set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100;				   

% set the number of nodes in output unit
n_class = 10;	

% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 150);

% set the regularization hyper-parameter
lambda = 0.0001;

% initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
%fprintf('\n Initialized weight W1');
initial_w2 = initializeWeights(n_hidden, n_class);
%fprintf('\n Initialized weight W2');

%Setting bias input unit to 1 for train, validation and test data 
train_data = [train_data ones(size(train_data,1),1)];
validation_data = [validation_data ones(size(validation_data,1),1)];
test_data = [test_data ones(size(test_data,1),1)];

% unroll 2 weight matrices into single column vector
initialWeights = [initial_w1(:); initial_w2(:)];

% define the objective function
%fprintf('\n Befor Calling objFunction');
objFunction = @(params) nnObjFunction(params, n_input, n_hidden, ...
                       n_class, train_data, train_label_mat, lambda);
%fprintf('\n After Calling objFunction');

% run neural network training with fmincg
%fprintf('\n Before Calling fmincg from Script.m');
[nn_params, cost] = fmincg(objFunction, initialWeights, options);
%fprintf('\n After Calling fmincg from Script.m');


% reshape the nn_params from a column vector into 2 matrices w1 and w2
w1 = reshape(nn_params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(nn_params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%fprintf('\n Reshaped w1 and w2 in Script.m');


%   Test the computed parameters
%fprintf('\n Before Calling nnPredict for train_data from Script.m');
predicted_label = nnPredict(w1, w2, train_data);
fprintf('\nTraining Set Accuracy: %f', ...
         mean(double(predicted_label == train_label)) * 100);
%fprintf('\n After Calling nnPredict for train_data from Script.m');     




%   Test Neural Network with validation data
%fprintf('\n Before Calling nnPredict for validation_data from Script.m');
predicted_label = nnPredict(w1, w2, validation_data);
fprintf('\nValidation Set Accuracy: %f', ...
         mean(double(predicted_label == validation_label)) * 100);
%fprintf('\n After Calling nnPredict for validation_data from Script.m');         

%   Test Neural Network with test data
%fprintf('\n Before Calling nnPredict for test_data from Script.m');
predicted_label = nnPredict(w1, w2, test_data);
fprintf('\nTesting Set Accuracy: %f', ...
      mean(double(predicted_label == test_label)) * 100);
%fprintf('\n After Calling nnPredict for test_data from Script.m');      

% timeSpent = toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% **************K-Nearest Neighbors***************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
 k = 5;
% % Test KNN with validation data
predicted_label = knnPredict(k, train_data, train_label, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
        mean(double(predicted_label == validation_label)) * 100);

% % Test KNN with test data
predicted_label = knnPredict(k, train_data, train_label, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
        mean(double(predicted_label == test_label)) * 100);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % *******Save the learned parameters *************************
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% timeSpent = toc
save('params.mat', 'n_input', 'n_hidden', 'w1', 'w2', 'lambda', 'k');

