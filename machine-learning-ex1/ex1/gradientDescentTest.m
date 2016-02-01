% test gradientDescent

% Copyright 2015 The MathWorks, Inc.

% preconditions

%% Test 1: test gradientDescent function 
[theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
theta(1, 1)
J_hist(1000)
%ans = 0.85426
assert(0.85426 == J_hist(1000))
assert(5.2148 == theta(1, 1))
assert(-0.5733 == theta(2, 1))