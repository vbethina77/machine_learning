% test computeCost

% Copyright 2015 The MathWorks, Inc.

% preconditions

%% Test 1: test computeCost function 
assert(computeCost( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2]) == 11.9450)
%%assert(computeCost( [1 2 3; 1 3 4; 1 4 5; 1 5 6], [7;6;5;4], [0.1;0.2;0.3]) == 7.0175)