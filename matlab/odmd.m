function omega = odmd(data,dt,tol)
%ODMD   Observable Dynamic Mode Decomposition
%   E = ODMD(data,dt,tol) returns a vector E containing the approximate
%   eigenvalues computed using observable dynamic mode decompostion, based
%   on the input observables vector data, time step dt, and tolerance tol.
%   The default value for tol is 1e-6.
%
%   See also mp, vqpe, uvqpe.

%   Reference:
%   Y. Shen, D. Camps, S. Darbha, A. Szasz, K. Klymko, D.B. Williams-Young,
%   N.M. Tubman, and R. Van Beeumen.  Estimating eigenenergies from quantum
%   dynamics: A unified noise-resilient measurement-driven approach, 2023.
%   https://doi.org/10.48550/arXiv.2306.01858

%% defaults
if nargin < 3, tol = 1e-6; end

%% Hankel matrix
k = length(data);
X = vec2hankel(data,floor(k/3)+1,ceil(2/3*k));

%% (shifted) data matrices
X1 = X(:,1:end-1);
X2 = X(:,2:end);

%% svd
[U,S,V] = svd(X1,'econ');

%% rank truncation
r = sum(diag(S) > tol*S(1));
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);

%% dmd
Atilde = U'*X2*V/S;
mu = eig(Atilde);
omega = log(mu)/dt;

end
