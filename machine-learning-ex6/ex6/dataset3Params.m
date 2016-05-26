function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%{
CValues = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigmaValues = [0.01 0.03 0.1 0.3 1 3 10 30]';
%error = zeros(CValues.size() * sigmaValues.size() , 3);
error = zeros(length(CValues) * length(sigmaValues) , 3);
count = 1;
for cval = 1:length(CValues)
    for sigVal = 1:length(sigmaValues)
        tempC = CValues(cval, 1);
        tempSigma = sigmaValues(sigVal, 1);
        model= svmTrain(X, y, CValues(cval, 1), @(x1, x2) gaussianKernel(x1, x2, sigmaValues(sigVal, 1)));

        predictions = svmPredict(model, Xval);

    
        error(count, 1) = CValues(cval, 1);
        error(count, 2) = sigmaValues(sigVal, 1);
        error(count, 3) = mean(double(predictions ~= yval));
        count = count + 1;
    end
end


[minVal minInd] = min(error);

minIndex = minInd(1,3);

C = error(minIndex, 1);

sigma = error(minIndex, 2);
%}
C = 1;

sigma = .1;
% =========================================================================

end
