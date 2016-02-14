%% Run tests by typing: 'test gradientDescent' in Octave

%!shared theta, thetaexpected
%! theta = 5e-05
%! thetaexpected = [5.21475; -0.57335];
%! [theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
%!assert(theta == thetaexpected, -eps);
%!assert(theta(1,1), 5.2148, -eps);
%!assert(theta, thetaexpected, -eps, tol);


%!shared tol
%! tol = 5e-05
%! J = computeCost( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2])
%!assert(J, 11.9450)



%% Run tests by typing: 'test gradientDescent' in Octave

%!shared theta, thetaexpected, J_hist
%! theta = 5e-05;
%! thetaexpected = [5.21475; -0.57335];
%! J_histexpected = [0.8543];
%! [theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
%!assert(theta, thetaexpected, 1e-4);
%!assert(J_hist(1000), J_histexpected, 1e-4);

%% Run tests by typing: 'test gradientDescent' in Octave

%!shared theta, thetaexpected, J_hist, J_histexpected;
%! theta = 5e-05;
%! thetaexpected = [5.21475; -0.57335];
%! J_histexpected = [0.85426];
%! J_hist = zeros(1000, 1);
%! [theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
%!assert(theta, thetaexpected, 1e-4);
%!assert(J_hist(1000), 0.85426, 1e-4);