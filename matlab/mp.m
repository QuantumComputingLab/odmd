function omega = mp(data,dt,tol)
%MP   Matrix Pencil Method
%   E = MP(data,dt,tol) returns a vector E containing the approximate
%   eigenvalues computed using the matrix pencil method, based on the
%   input observables vector data, time step dt, and tolerance tol.
%   The default value for tol is 1e-6.
%
%   See also odmd, vqpe, uvqpe.

%   Reference:
%   T.K. Sarkar and O. Pereira.  Using the matrix pencil method to estimate
%   the parameters of a sum of complex exponentials, IEEE Antennas and
%   Propagation Magazine 37, pp.48-55, 1995.
%   https://doi.org/10.1109/74.370583

%% defaults
if nargin < 3, tol = 1e-6; end

%% Hankel matrix
k = length(data);
X = vec2hankel(data,floor(k/3)+1,ceil(2/3*k));

%% svd
[~,S,V] = svd(X,'econ');

%% rank truncation
r = sum(diag(S) > tol*S(1));
V = V(:,1:r);
V1 = V(1:end-1,:);
V2 = V(2:end,:);

%% mp
Atilde = pinv(V2)*V1;
mu = eig(Atilde);
omega = log(mu)/dt;

end
