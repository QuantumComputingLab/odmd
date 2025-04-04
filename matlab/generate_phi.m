function phi = generate_phi(overlap,N)
%GENERATE_PHI   Initial state with overlap
%   phi = GENERATE_PHI(overlap,N) returns a vector of lenght N such that
%
%     |phi'*gs|^2 = overlap,
%
%   where gs is the ground state. The overlap of phi with the remaining
%   eigenstates is set to be uniformly equal.
%
%   See also generate_samples.

%% starting vector
phi = zeros(N,1);
phi(1) = sqrt(overlap);
phi(2:end) = sqrt((1 - phi(1)^2)/(N - 1));

end
