function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    hofx = X*theta;
    difference = hofx - y;
    
    temp = transpose(X) * difference;
    %sumofDifference = sum(difference);
    %difference
    %X(:,2)
    %diffX = sum(difference*X(1,2)); 
    %theta0 = theta0 - (alpha/m) * sumofDifference;
    %theta1 = theta1 - ((alpha/m) * diffX);
     %theta0
     %theta1
     
    %theta0 = theta0 - (alpha/m) * temp;
    %theta1 = theta1 - (alpha/m) * temp;
    %theta(1,1) = theta0;
    %theta(2,1) = theta1;
    theta = theta - (alpha/m) * temp;
    theta








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
