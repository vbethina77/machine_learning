function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

temp = sigmoid(X*transpose(all_theta));
%size(all_theta)
%size(X)
%temp = sigmoid(all_theta*transpose(X));
%temp
%p = max(temp, [], 2)
%maxarray = max(temp, [], 2);
%maxarray
%for i=1:num_labels
%    p(maxarray <= .5) = 0;
%    p(maxarray > .5) = num_labels;
%end
temp
sigmoid(X*all_theta')
[values, indexes] = max(sigmoid(X*all_theta'), [], 2);
values
p = indexes;

% =========================================================================


end

%% Define unit tests, see also https://www.gnu.org/software/octave/doc/interpreter/Test-Functions.html
%% Can be run by typing: 'test predictOneVsAll' in Octave

%!shared output, expected
%! all_theta = [1 -6 3; -2 4 -3];
%! X = [1 7; 4 5; 7 8; 1 4];
%! expected = [1; 2; 2; 1]
%! output = predictOneVsAll(all_theta, X)
%!assert(output, expected)
