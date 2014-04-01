function [obj_val obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, training_data,...
                                    training_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.

w1 = reshape(params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%fprintf('\nnObjFunction started.');
%Initialising variables
grad_w1 = zeros(n_hidden,n_input + 1);
grad_w2 = zeros(n_class,n_hidden + 1);
w1MatValue = 0;
w2MatValue = 0;
errorValue = 0;


%%%%%%%%%%%%%%%%%Feed Forward Network%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Adding a bias hidden unit 
sigmoidHidden = zeros(size(training_data,1),n_hidden+1);
sigmoidHidden(:,(n_hidden+1)) = 1;

%Activated values of hidden units
sigmoidHidden = sigmoid(training_data*w1');
sigmoidHidden = [sigmoidHidden ones(size(sigmoidHidden,1),1)];

%Output values
sigmoidOut = sigmoid(sigmoidHidden*w2');



%%%%%%%%%%%%%%%%%Back Propagation Logic%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Negative likelihood error value for each image
errorValueImg = (training_label.*log(sigmoidOut)) + ((1-training_label).*log(1-sigmoidOut));

%Cumulative sum of error value
errorValue = sum(errorValueImg(:));
N = size(training_data,1);
errorValue =(-1)*(errorValue/N);

w1MatValue = w1.^2;
w2MatValue = w2.^2;

w1MatValue = sum(w1MatValue(:));
w2MatValue = sum(w2MatValue(:));

%Final scalar value of error function with regularization parameter lambda
errorValue = errorValue + ((lambda/(2*N))*(w1MatValue + w2MatValue));

%Derivative of error function wrt weights from hidden units to output units
deltaK = sigmoidOut - training_label;
grad_w2 = deltaK'*sigmoidHidden;
grad_w2 = grad_w2 + (lambda*w2);
grad_w2 = (grad_w2)/N;

%Removing the hidden bias unit
sigmoidHidden = sigmoidHidden(:,1:(end-1));
w2 = w2(:,1:(end-1));

%Derivative of error function wrt weights from input units to hidden units
summationDelta = deltaK*w2;
grad_w1 = (1 - sigmoidHidden).*(sigmoidHidden).*summationDelta;
grad_w1 = grad_w1'*training_data;
grad_w1 = grad_w1 + (lambda*w1);
grad_w1 = (grad_w1)/N;

%fprintf('\nBefore ending nnObjFunction');

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 andl grad_w2 
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];
obj_val = errorValue;
end
