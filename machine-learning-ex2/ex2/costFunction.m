function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%
%J = sum(-y*log(sigmoid(X))-(1-y)*log(1-sigmoid(X)));
sigmoidfn = (sigmoid(X*theta));
temp=transpose(log(sigmoidfn))*y + transpose(log(1-sigmoidfn))*(1-y);
J = -1*temp/m;

grad=(transpose(X)*(sigmoidfn-y))/m;




% =============================================================

end

%% Define unit tests, see also https://www.gnu.org/software/octave/doc/interpreter/Test-Functions.html
%% Can be run by typing: 'test costFunction' in Octave

%!shared tol
%! tol = 5e-05
%! X = [ones(3,1) magic(3)];
%! y = [1 0 1]';
%! theta = [-2 -1 1 2]';
%! gresult = [0.31722 0.87232 1.64812 2.23787]';
%! [j g] = costFunction(theta, X, y)
%!assert (j = 4.6832)
%!assert (g(1,1) = 0.31722)
%!assert (g(2,1) = 0.87232)
%!assert (g(3,1) = 1.64812)
%!assert (g(4,1) = 2.23787)



