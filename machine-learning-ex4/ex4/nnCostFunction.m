function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
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
                 num_labels, (hidden_layer_size + 1));

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

%Reference materail
%https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/QFnrpQckEeWv5yIAC00Eog

% -------------------------------------------------------------

%1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). This is most easily done using an eye() matrix of size num_labels, with vectorized indexing by 'y'. A useful variable name would be "y_matrix", as this...


eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m,1) X];
%z2 equals the product of a1 and ?1
z2=a1*Theta1';
%a2 is the result of passing z2 through g()
a2=sigmoid(z2);
%Then add a column of bias units to a2 (as the first column).
a2 = [ones(size(a2, 1), 1) a2];
%NOTE: Be sure you DON'T add the bias units as a new row of Theta.

%z3 equals the product of a2 and ?2
z3=a2*Theta2';
%a3 is the result of passing z3 through g()
a3=sigmoid(z3);

%3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
%using a3, your y_matrix, and m (the number of training examples).
%Note that the 'h' argument inside the log() function is exactly a3. 
%Cost should be a scalar value. Since y_matrix and a3 are both matrices, you need to compute the double-sum.
%Remember to use element-wise multiplication with the log() function. Also, we're using the natural log, not log10().
temp = -y_matrix.*log(a3) - (1-y_matrix).*log(1-a3);
J = sum(temp(:)) /m;



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
d3 = a3 - y_matrix;
%3: z2 came from the forward propagation process - it's the product of a1 and Theta1, 
%   prior to applying the sigmoid() function. Dimensions are (m x n) ? (n x h) --> (m x h)

%4: ?2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 and Theta2(no bias), 
%   then element-wise scaled by sigmoid gradient of z2. The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2, as must be.
%d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);
d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);
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
%Q = rand(3,4)       % create a test matrix
%Q(:,1) = 0          % set the 1st column of all rows to 0
%9: Scale each Theta matrix by ?/m. Use enough parenthesis so the operation is correct.

%10: Add each of these modified-and-scaled Theta matrices to the un-regularized Theta gradients that you computed earlier.

%You're done. Use the test case (linked below) to test your code, and the ex4 script, then run the submit script.
%{
for t = 1:m
  a_1 = X(t,:);
  %ones(m,1) X
  %a2 = [ones(size(a2, 1), 1) a2];
  a1_withBias = [ones(size(a_1, 1), 1) a_1];
  z_2 = Theta1*a1_withBias';
  a_2 = sigmoid(z_2);
  %a2_withBias = [1 ;  a_2];
  z_3 = Theta2*[1; a_2];
  a_3 = sigmoid(z_3);
  
  % come back and review this as a_3 may not be what we need?
  error_3 = a_3 - y_matrix(t,:)';
  
  % step 3 for hidden layre l = 2 calcualte error
  error_2 = (Theta2'*error_3).*[1; sigmoidGradient(z_2)];
   
  %step 4 accumulate delte
  %___delta_2 = delta_2 + error_2*transpose(a_3);
  
  Theta2_grad = Theta2_grad + error_3*[1; a_2]';
  error_2 = error_2(2: end);
  Theta1_grad = Theta1_grad + error_2*a1_withBias;
  
end

Theta1_grad = (1/m)*Theta1_grad + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
%}
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
