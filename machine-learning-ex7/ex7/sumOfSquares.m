Q = magic(3);
v = [1 2 3; 2 4 5];
for i = 1 : size(v, 1)
 temp =  v(i, :);
 diff = bsxfun(@minus, Q, temp);
 sum(diffs.^2, 2)
end
%sum(diffs.^2, 2)