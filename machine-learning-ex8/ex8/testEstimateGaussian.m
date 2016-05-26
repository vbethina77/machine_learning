%Test 1a (Estimate Gaussian Parameters):
%input:
X = sin(magic(4));
X = X(:,1:3);
[mu sigma2] = estimateGaussian(X)
%output:
%fprintf('Mean', mu);
%fprintf('Best F1 on Cross Validation Set:  %f\n', sigma2);
%{
mu =
  -0.3978779  0.3892253  -0.0080072
sigma2 =
   0.27795    0.65844   0.20414
%}