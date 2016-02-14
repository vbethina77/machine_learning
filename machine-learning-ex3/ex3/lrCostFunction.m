function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%sigmoidfn = (sigmoid(X*theta));
%temp=transpose(log(sigmoidfn))*y + transpose(log(1-sigmoidfn))*(1-y);
%J = -1*temp/m;

%grad=(transpose(X)*(sigmoidfn-y))/m;

sigmoidfn = (sigmoid(X*theta));
temp=transpose(log(sigmoidfn))*y + transpose(log(1-sigmoidfn))*(1-y);
modifiedThetaVector = theta(2 : size(theta));
regpart = (lambda * sum(modifiedThetaVector.^2))/ (2*m);
J = (-1*temp/m) + regpart;


modifiedThetaForGrad = theta;
modifiedThetaForGrad(1,1) = 0;
%modifiedThetaForGrad
regpartgrad = (lambda * modifiedThetaForGrad)/m;
grad=((transpose(X)*(sigmoidfn-y))/m) + regpartgrad;






% =============================================================

grad = grad(:);

end

%% Define unit tests, see also https://www.gnu.org/software/octave/doc/interpreter/Test-Functions.html
%% Can be run by typing: 'test lrCostFunction' in Octave

%!shared expectedGrad, grad, J
%! theta = [-2; -1; 1; 2];
%! X = [ones(3,1) magic(3)];
%! y = [1; 0; 1] >= 0.5;       % creates a logical array
%! lambda = 3;
%! expectedGrad = [0.31722; -0.12768;  2.64812; 4.23787];
%! [J grad] = lrCostFunction(theta, X, y, lambda);
%!assert(J, 7.6832, 1e-4);
%!assert(grad, expectedGrad, 1e-4);
