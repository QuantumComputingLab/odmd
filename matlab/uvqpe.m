function omega = uvqpe(data,dt,tol)
%UVQPE   Unitary Variational Quantum Phase Estimation
%   E = UVQPE(data,dt,tol) returns a vector E containing the approximate
%   eigenvalues computed using unitary variational quantum phase estimation,
%   based on the input observables vector data, time step dt, and tolerance tol.
%   The default value for tol is 1e-6.
%
%   See also vqpe, odmd, mp.

%   Reference:
%   K. Klymko, C. Mejuto-Zaera, S.J. Cotton, F. Wudarski, M. Urbanek, D. Hait,
%   M. Head-Gordon, K.B. Whaley, J. Moussa, N. Wiebe, W.A. de Jong, and N.M.
%   Tubman.﻿ Real-time evolution for ultracompact Hamiltonian eigenstates on
%   quantum hardware, PRX Quantum 3, 020323, 2022.
%   https://doi.org/10.1103/PRXQuantum.3.020323

%% defaults
if nargin < 3, tol = 1e-6; end

%% Toeplitz matrices
if length(data) == 2
    H = data(2);
else
    H = toeplitz([data(2);data(1);conj(data(2:end-2))],data(2:end));
end
S = toeplitz(data(1:end-1));

%% rank truncation
[V,d] = eig(S,'vector');
V = V(:,end:-1:1);
d = d(end:-1:1);
r = sum(d > tol*d(1));
V = V(:,1:r);

%% eigenvalues
Ht = V'*H*V;
St = diag(d(1:r));
mu = eig(Ht,St);
omega = log(mu)/dt;

end
