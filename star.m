kernsp = kernel('lap','sprime');

% get a chunker discretization of a starfish domain
chnkr = chunkerfunc(@(t) starfish(t), struct('nchmin', 64));
% struct('nchmin', 31), struct('nchmax', 33)

% define a boundary condition. because of the symmetries of the 
% starfish, this function integrates to zero
pwfun = @(r) r(1,:).^2.*r(2,:);
rhs = pwfun(chnkr.r); rhs = rhs(:);

% get the kernel 
kernsp = kernel('lap','sprime');

% get a matrix discretization of the boundary integral equation 
sysmat = chunkermat(chnkr,kernsp); % just the sprime part
% add the identity term and "ones matrix"
sysmat = sysmat + 0.5*eye(chnkr.npt);
writematrix(sysmat, 'starfish.csv');