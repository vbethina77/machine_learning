function [theta, J_history] = gradientDescent_vectorized(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %

    hofx = X*theta;
    difference = hofx - y;
    
    temp = transpose(X) * difference;
    theta = theta - (alpha/m) * temp;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end


%% Run tests by typing: 'test gradientDescent_vectorized' in Octave

%!shared theta, thetaexpected, J_hist
%! thetaexpected = [5.21475; -0.57335];
%! [theta J_hist] = gradientDescent_vectorized([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
%!assert(theta, thetaexpected, 1e-4);
%!assert(J_hist(1000), 0.85426, 1e-4);
