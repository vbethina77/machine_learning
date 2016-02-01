% test sigmoid

% Copyright 2015 The MathWorks, Inc.

% preconditions
% angles = rightTri(tri);
% assert(angles(3) == 90,'Fundamental problem: rightTri not producing right triangle')

%% Test 1: test sigmoid function 
assert(sigmoid(1200000) == 1)
assert(sigmoid(-25000) == 0) 
assert(sigmoid(0) == 0.5)
temp = sigmoid([4 5 6]) == [0.9820 0.9933 0.9975];
assert(temp(1, 1)== 0)
assert(temp(1, 2)== 0)
assert(temp(1, 3)== 0)