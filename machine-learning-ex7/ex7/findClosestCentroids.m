function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
%{
distanceMatrix = zeros(length(X), K);
for i = 1 : length(X)
    for centroid = 1 : K;
        xValue = X(i,:);
        centroidValue = centroids(centroid,:);
        distance = X(i,:) - centroids(centroid,:);
        sqrDistance = distance.^2;
        distanceMatrix(i, centroid) = sqrDistance;
    end
end

[Y,I] = min(distanceMatrix, [], 2);

idx = I;
%}
%Create a "distance" matrix of size (m x K) and initialize it to all zeros. 'm' 
%is the number of training examples, K is the number of centroids.
m = size(X, 1);
distance = zeros(m, K);

%Use a for-loop over the 1:K centroids.
for centroid = 1 : K
    %Inside this loop, create a column vector of the distance from each training example to that centroid, 
    %and store it as a column of the distance matrix. One method is to use the sum() function and the bsxfun() 
    %function to calculate the sum of the squares of the differences between each row in the X matrix and a centroid.
    diff = bsxfun(@minus, X,  centroids(centroid,:));
    distance(:, centroid)  = sum(diff.^2, 2)';
end

%When the for-loop ends, you'll have a matrix of centroid distances.
%Then return idx as the vector of the indexes of the locations with the minimum distance. The result is a vector of size (m x 1) with the indexes of the closest centroids.
[Y,I] = min(distance, [], 2);

idx = I;
% =============================================================

end

