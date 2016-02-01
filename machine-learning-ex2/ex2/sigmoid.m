function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
temp = 1+exp(-z);


g = temp.^-1;


% =============================================================

end


%% Define unit tests, see also https://www.gnu.org/software/octave/doc/interpreter/Test-Functions.html
%% Can be run by typing: 'test sigmoid' in Octave

%!assert (sigmoid(1200000), 1)
%!assert (sigmoid(-25000), 0)
%!assert (sigmoid(0), 0.5)

%!shared tol
%! tol = 5e-05
%!assert (sigmoid([4 5 6]), [0.9820 0.9933 0.9975], tol)
%!assert (sigmoid(magic(3)), [0.9997 0.7311 0.9975; 0.9526 0.9933 0.9991; 0.9820 0.9999 0.8808], tol)
%!assert (sigmoid(eye(2)), [0.7311 0.5000; 0.5000 0.7311], tol)
