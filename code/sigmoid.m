function g = sigmoid(z)
% sigmoid computes sigmoid functoon
% Notice that z can be a scalar, a vector or a matrix

g = 1./(1+exp(-z));

end
