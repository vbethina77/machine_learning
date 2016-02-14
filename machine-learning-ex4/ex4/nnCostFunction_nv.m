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


%sigmoidfn = (sigmoid(X*theta));
%temp=transpose(log(sigmoidfn))*y + transpose(log(1-sigmoidfn))*(1-y);
%modifiedThetaVector = theta(2 : size(theta));
%regpart = (lambda * sum(modifiedThetaVector.^2))/ (2*m);
%J = (-1*temp/m) + regpart;



size(Theta1)
size(Theta2)
%size(X)

Activation1 = [ones(m, 1) X];
size(Activation1)
z2 = Activation1*Theta1';

Activation2 = sigmoid(z2);
Activation2 = [ones(size(Activation2, 1), 1) Activation2];
z3 = Activation2*Theta2';

Activation3 = sigmoid(z3);



%[values, indexes] = max(Activation3, [], 2);
%p = indexes;   
%size(p); 
%temp=-transpose(log(p))*y - transpose(log(1-p))*(1-y);

%J=temp/m;

%{
yy = zeros(size(y, 1),num_labels);
for i=1:size(X)
  yy(i,y(i)) = 1;
end
%}

%Create a y matrix where each row represents a 1 vs all entry based on num
% labels
 ytransformed = bsxfun(@eq, y, 1:num_labels);
size(ytransformed);
%ytransformed

X = [ones(m,1) X];

for  i=1:m
  a1 = X(i,:);
  z2 = Theta1*a1';
  a2 = sigmoid(z2);
  z3 = Theta2*[1; a2];
  a3 = sigmoid(z3);
  J = J + -ytransformed(i,:)*log(a3)-(1-ytransformed(i,:))*log(1-a3);
end

J = J/m;

% Regularized Part
modifiedTheta1 = Theta1(:,2:end);
modifiedTheta2 = Theta2(:,2:end);
sqrT1 = modifiedTheta1.^2;
sqrT2 = modifiedTheta2.^2;

sumof =sum(sqrT1(:)) + sum(sqrT2(:));

regpart = (lambda*sumof)/(2*m);

J = J + regpart;
%{
ytransize = size(ytransformed);
activationsize = size(Activation3);
J = -ytransformed'*log(Activation3)-(1-ytransformed)'*log(1-Activation3);
size(J)
sumofJ = sum(J)
size(sumofJ)
J = sumofJ/m;
%}

for t = 1:m
  a_1 = X(t,:);
  %a1_withBias = [1 ; a_1];
  z_2 = Theta1*a_1';
  a_2 = sigmoid(z_2);
  %a2_withBias = [1 ;  a_2];
  z_3 = Theta2*[1; a_2];
  a_3 = sigmoid(z_3);
  
  % come back and review this as a_3 may not be what we need?
  error_3 = a_3 - ytransformed(t,:)';
  
  % step 3 for hidden layre l = 2 calcualte error
  error_2 = (Theta2'*error_3).*[1; sigmoidGradient(z_2)];
   
  %step 4 accumulate delte
  %___delta_2 = delta_2 + error_2*transpose(a_3);
  
  Theta2_grad = Theta2_grad + error_3*[1; a_2]';
  error_2 = error_2(2: end);
  Theta1_grad = Theta1_grad + error_2*a_1;
  
end

Theta1_grad = (1/m)*Theta1_grad + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
