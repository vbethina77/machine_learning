function [J grad] = nnCostFunctionLinear(nn_params, ...
                                         input_layer_size, ...
                                         hidden_layer_size, ...
                                         X, y, lambda)

                                     %NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% =========================================================================

%a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m,1) X];
%z2 equals the product of a1 and ?1
z2=a1*Theta1';
%a2 is the result of passing z2 through g()
a2=tanhz(z2);
%Then add a column of bias units to a2 (as the first column).
a2 = [ones(size(a2, 1), 1) a2];
%NOTE: Be sure you DON'T add the bias units as a new row of Theta.

%z3 equals the product of a2 and ?2
z3=a2*Theta2';
%a3 is the result of passing z3 through g()
a3=z3;

%3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
%using a3, your y_matrix, and m (the number of training examples).
%Note that the 'h' argument inside the log() function is exactly a3. 
%Cost should be a scalar value. Since y_matrix and a3 are both matrices, you need to compute the double-sum.
%Remember to use element-wise multiplication with the log() function. Also, we're using the natural log, not log10().
%temp = -y_matrix.*log(a3) - (1-y_matrix).*log(1-a3);
%J = sum(temp(:)) /m;

hypothesis = a3;
difference = hypothesis - y;
square = difference.^2;
sumofsquare = sum(square);
J = sumofsquare/(2*m);

%Regularized part
modifiedTheta1 = Theta1(:,2:end);
modifiedTheta2 = Theta2(:,2:end);
sqrT1 = modifiedTheta1.^2;
sqrT2 = modifiedTheta2.^2;

sumof =sum(sqrT1(:)) + sum(sqrT2(:));

regpart = (lambda*sumof)/(2*m);

J = J + regpart;


% Back Propogation
% https://www.coursera.org/learn/machine-learning/discussions/a8Kce_WxEeS16yIACyoj1Q
%1: Perform forward propagation, similarly to how it is explained in this tutorial:

%https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/QFnrpQckEeWv5yIAC00Eog.

%2: ?3 or d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
d3 = a3 - y;
%3: z2 came from the forward propagation process - it's the product of a1 and Theta1, 
%   prior to applying the sigmoid() function. Dimensions are (m x n) ? (n x h) --> (m x h)

%4: ?2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 and Theta2(no bias), 
%   then element-wise scaled by sigmoid gradient of z2. The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2, as must be.
%d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);
d2 = d3*Theta2(:,2:end).*tanhGradient(z2);
%5: ?1 or Delta1 is the product of d2 and a1. The size is (h x m) ? (m x n) --> (h x n)
Delta1 = d2'*a1;
%6: ?2 or Delta2 is the product of d3 and a2. The size is (r x m) ? (m x [h+1]) --> (r x [h+1])
Delta2 = d3'*a2;
%7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;
%Now you have the unregularized gradients. Check your results using ex4.m, and submit this portion to the grader.

%===== Regularization of the gradient ===========

%Since Theta1 and Theta2 are local copies, and we've already computed our hypothesis value during forward-propagation, 
%we're free to modify them to make the gradient regularization easy to compute.

%8: So, set the first column of Theta1 and Theta2 to all-zeros. Here's a method you can try in your workspace console:
Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end