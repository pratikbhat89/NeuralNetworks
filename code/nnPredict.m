function label = nnPredict(w1, w2, data)
% nnPredict predicts the label of data given the parameter w1, w2 of Neural
% Network.

% Input:
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image
       
% Output: 
% label: a column vector of predicted labels


%fprintf('\nStarted nnPredict.');

%Creating label vector
label = zeros(size(data,1),1);

%Activated values of hidden units
sigmoidHidden = sigmoid(data*w1');

sigmoidHidden = [sigmoidHidden ones(size(sigmoidHidden,1),1)];

%Output values
sigmoidOut = sigmoid(sigmoidHidden*w2');

[C,I] = max(sigmoidOut,[],2);

%Predicting the label for each image
for i=1:size(data,1)
    label(i,1) = I(i)-1;
end    

%fprintf('\nEnded nnPredict.');
end
    
