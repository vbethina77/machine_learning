function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
sizeOfX = size(X)
sizeOfTheta = size(theta)
sigmoidValue = sigmoid(X*theta);

% Logical indexing approach
% http://www.mathworks.com/matlabcentral/answers/8817-how-to-replace-the-elements-of-a-matrix-using-the-conditions-if-else
p(sigmoidValue > .5) = 1;
p(sigmoidValue <= .5) = 0;


% =========================================================================


end

%% Define unit tests, see also https://www.gnu.org/software/octave/doc/interpreter/Test-Functions.html
%% Can be run by typing: 'test predict' in Octave

%! prediction = predict([0.3 ; 0.2], [1 2.4; 1 -17; 1 0.5])
%!assert (prediction(1,1) = 1)
%!assert (prediction(2,1) = 1)
%!assert (prediction(3,1) = 1)