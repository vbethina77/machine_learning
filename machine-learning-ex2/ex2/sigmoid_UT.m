if(sigmoid(1200000) == 1)
    fprintf('\n Pass\n');
else
    fprintf('\n Fail\n');
end;

if(sigmoid(-25000) == 0)
    fprintf('\n Pass\n');
else
    fprintf('\n Fail\n');
end;

if(sigmoid(0) == 0.5)
    fprintf('\n Pass\n');
else
    fprintf('\n Fail\n');
end;

sigmoid([4 5 6])
%ans = 0.9820 0.9933 0.9975

