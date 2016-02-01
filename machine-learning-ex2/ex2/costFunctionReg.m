function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigmoidfn = (sigmoid(X*theta));
temp=transpose(log(sigmoidfn))*y + transpose(log(1-sigmoidfn))*(1-y);
modifiedThetaVector = theta(2 : size(theta));
regpart = (lambda * sum(modifiedThetaVector.^2))/ (2*m);
J = (-1*temp/m) + regpart;


modifiedThetaForGrad = theta;
modifiedThetaForGrad(1,1) = 0;
modifiedThetaForGrad
regpartgrad = (lambda * modifiedThetaForGrad)/m;
grad=((transpose(X)*(sigmoidfn-y))/m) + regpartgrad;





% =============================================================

end
