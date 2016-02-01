function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta0 = theta(1,1);
theta1 = theta(2,1);

numberOfFeatures = size(X, 2)
%theta = zeros(numberOfFeatures, 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    hofx = X*theta;
    difference = hofx - y;
    for j = 1:numberOfFeatures
        sumOfValues=0;

        for i = 1:m
           sumOfValues = sumOfValues + difference(i, 1)*X(i, j);
        end
        theta(j, 1) = theta(j, 1) - (alpha * sumOfValues) / m;
    end
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
