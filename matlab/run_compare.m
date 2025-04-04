function [lam,t] = run_compare(dataH,dataS,dt,fun,tol,Tmax,step)

%% defaults
if nargin < 4; fun = @odmd; end
if nargin < 5; tol = [1e-1,1e-2,1e-3]; end
if nargin < 6; Tmax = 500; end
if nargin < 7; step = 10; end

%% eigenvalue approximation
t = 2:step:Tmax;
lam = inf(length(t),length(tol));
for i = 1:length(t)
    for j = 1:length(tol)
        if isequal(fun,@vqpe)
            omega = fun(dataH(1:t(i)),dataS(1:t(i)),tol(j));
            assert(isreal(omega));
            lam(i,j) = min(omega);
        else
            omega = fun(dataS(1:t(i)),dt,tol(j));
            lam(i,j) = -max(imag(omega));
        end
    end
end

end
