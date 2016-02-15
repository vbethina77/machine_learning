%% Initialization
clear ; close all; clc

% inputs
%nn_params = [31 16 15 -29 -13 -8 -7 13 54 -17 -11 -9 16]'/ 10;
il = 1;
hl = 4;
X = [1 ; 2 ; 3];
y = [1 ; 4 ; 9];
lambda = 0.01;

Theta1 = randInitializeWeights(1, 4);

Theta2 = randInitializeWeights(4, 1);

nn_params = [Theta1(:) ; Theta2(:)];

% command
[j g] = nnCostFunctionLinear(nn_params, il, hl, X, y, lambda);

% results
j_out =  0.020815

%{
g =
    -0.0131002
    -0.0110085
    -0.0070569
     0.0189212
    -0.0189639
    -0.0192539
    -0.0102291
     0.0344732
     0.0024947
     0.0080624
     0.0021964
     0.0031675
    -0.0064244
%}