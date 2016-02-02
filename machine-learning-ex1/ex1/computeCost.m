function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%J = (sum((transpose(theta)*X-y)^2))/2m;

hypothesis = X*theta;
difference = hypothesis - y;
square = difference.^2;
sumofsquare = sum(square);
J = sumofsquare/(2*m);

% =========================================================================

end

%% Define unit tests, see also https://www.gnu.org/software/octave/doc/interpreter/Test-Functions.html
%% Can be run by typing: 'test computeCost' in Octave

%!assert( computeCost( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2] )
%!        ,11.9450, -eps )
%!assert( computeCost( [1 2 3; 1 3 4; 1 4 5; 1 5 6], [7;6;5;4],
%!        [0.1;0.2;0.3] ), 7.0175, -eps )
