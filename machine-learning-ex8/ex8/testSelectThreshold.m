%Test 2a (Select threshold):
%input:
[epsilon F1] = selectThreshold([1 0 0 1 1]', [0.1 0.2 0.3 0.4 0.5]');
%output:%epsilon =  0.40040
%F1 =  0.57143
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
