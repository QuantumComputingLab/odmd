function H = vec2hankel(data,m,n)
%VEC2HANKEL   Convert vector into Hankel matrix
%   H = VEC2HANKEL(data,m,n) returns an m x n Hankel matrix whose first column
%   is data(1:m) and whose last row is data(m:m+n-1).
%
%         [ d1   d2   d3   ... dn     ]
%         [ d2   d3   d4   ... dn+1   ]
%     H = [ d3   d4   d5   ... dn+2   ]
%         [  :    :    :    .   :     ]
%         [ dm   dm+1 dm+2 ... dm+n-1 ]
%
%   See also hankel.

%% defaults
if nargin < 3, n = m; end

%% check
assert(length(data) >= m+n-1);

%% Hankel matrix
H = hankel(data(1:m),data(m:m+n-1));

end
