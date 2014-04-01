function [train_data, train_label_vector,train_label_mat, validation_data, ...
    validation_label_vector,test_data, test_label_vector] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of training set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

%fprintf('\nBefore Loading data');
load('mnist_all.mat')

%Combining all training and testing data
 trainingset = [train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
 testingset  = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9];

% Creating labels for training data
trainlabel_0 = ones(size(train0,1),1) * 0;
trainlabel_1 = ones(size(train1,1),1) * 1;
trainlabel_2 = ones(size(train2,1),1) * 2;
trainlabel_3 = ones(size(train3,1),1) * 3;
trainlabel_4 = ones(size(train4,1),1) * 4;
trainlabel_5 = ones(size(train5,1),1) * 5;
trainlabel_6 = ones(size(train6,1),1) * 6;
trainlabel_7 = ones(size(train7,1),1) * 7;
trainlabel_8 = ones(size(train8,1),1) * 8;
trainlabel_9 = ones(size(train9,1),1) * 9;

trainlabel_data = [trainlabel_0;trainlabel_1;trainlabel_2;trainlabel_3;trainlabel_4;trainlabel_5;trainlabel_6;trainlabel_7;trainlabel_8;trainlabel_9];

% Creating labels for testing data
testlabel_0 = ones(size(test0,1),1) * 0 ;
testlabel_1 = ones(size(test1,1),1) * 1 ;
testlabel_2 = ones(size(test2,1),1) * 2 ;
testlabel_3 = ones(size(test3,1),1) * 3 ;
testlabel_4 = ones(size(test4,1),1) * 4 ;
testlabel_5 = ones(size(test5,1),1) * 5 ;
testlabel_6 = ones(size(test6,1),1) * 6 ;
testlabel_7 = ones(size(test7,1),1) * 7 ;
testlabel_8 = ones(size(test8,1),1) * 8 ;
testlabel_9 = ones(size(test9,1),1) * 9 ;


testlabel_data = [testlabel_0;testlabel_1;testlabel_2;testlabel_3;testlabel_4;testlabel_5;testlabel_6;testlabel_7;testlabel_8;testlabel_9];

%Assigning the labels to train and test data
train_data_labeled = double([trainingset trainlabel_data]);
test_data_labeled = double([testingset testlabel_data]);

%Randomize all the rows to split the 60000 training set in 50000 train set and 10000 validation set
trrows = size(train_data_labeled,1);
randRows_trn = randperm(trrows);
train_data_rand = train_data_labeled(randRows_trn(1:50000),:);
validation_data_rand = train_data_labeled(randRows_trn(50001:end),:);

testrows = size(test_data_labeled,1);
randRows_test = randperm(testrows);
test_data_rand = test_data_labeled(randRows_test,:);

%Separating the data and its label values
train_data = mat2gray(train_data_rand(:,1:784));
train_label_vector = (train_data_rand(:,785));
validation_data = mat2gray(validation_data_rand(:,1:784));
validation_label_vector= (validation_data_rand(:,785));
test_data = mat2gray(test_data_rand(:,1:784));
test_label_vector = (test_data_rand(:,785));

%Generating 1 of K coding scheme vector for every image
train_label_mat = zeros(size(train_label_vector,1),10);


for i = 1:size(train_label_vector,1)
    idx = train_label_vector(i,1);
    train_label_mat(i,idx+1)=1;
end


%fprintf('\nAfter Loading data');

