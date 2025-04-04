
%% parameters
molecule = 'Cr2';
noise = 0.001;
overlap = 0.05;   % 0: HF
fun = @odmd;      % @odmd, @mp, @vqpe, @uvqpe
tol = 0.01;       % relative truncation error for SVD/EIG
Tmax = 200;       % total time steps
step = 5;         % step size for plotting
datatype = 'c';   % r: real, i: imaginary, c: complex
noisetype = 'c';  % r: real, i: imaginary, c: complex
savedata = 0;

%% fix random seed
rng(100);

%% Cr2 data (true eigenvalues + HF ground state)
if strcmp(molecule,'Cr2')
    load('Cr2_4000.mat');
elseif strcmp(molecule,'LiH')
    load('LiH_2989.mat');
elseif strcmp(molecule,'H6')
    load('H6_200.mat');
else
    error('Wrong molecule!');
end
Et = lam2lamt(E,E(1),E(end));  % convert to [-pi/4,pi/4]
dt = 3;
if overlap == 0
    [dataH,dataS] = generate_samples(Et,psiHF,dt,Tmax);
else
    phi = generate_phi(overlap,length(E));
    [dataH,dataS] = generate_samples(Et,phi,dt,Tmax);
end

%% add noise
if strcmpi(noisetype,'r') || strcmpi(noisetype,'c')
    dataH = dataH + noise*randn(Tmax,1);
    dataS = dataS + noise*[0;randn(Tmax-1,1)];
elseif strcmpi(noisetype,'i') || strcmpi(noisetype,'c')
    dataH = dataH + 1i*noise*[0;randn(Tmax-1,1)];
    dataS = dataS + 1i*noise*[0;randn(Tmax-1,1)];
end
if strcmpi(datatype,'r')
    dataH = real(dataH);
    dataS = real(dataS);
elseif strcmpi(datatype,'i')
    dataH = imag(dataH);
    dataS = imag(dataS);
end

%% run
[lamt,t] = run_compare(dataH,dataS,dt,fun,tol,Tmax,step);
lam = lamt2lam(lamt,E(1),E(end));  % convert back from [-pi/4,pi/4]

%% save
if savedata
    if overlap == 0
        filename = '-HF';
    else
        filename = ['-o',num2str(100*overlap)];
    end
    filename = [molecule,'-n',num2str(noise),filename,'-',func2str(fun),'.dat'];
    writematrix([t(:) abs(lam - E(1))],filename,'Delimiter','\t');
end

%% plot
mytitle = [molecule,' (noise = ',num2str(noise)];
if overlap == 0, mytitle = [mytitle,'  -  HF)'];
else,            mytitle = [mytitle,'  -  overlap = ',num2str(overlap),')']; end
if isequal(fun,@odmd),      mytitle = ['ODMD: ',mytitle];
elseif isequal(fun,@mp),    mytitle = ['Matrix Pencil: ',mytitle];
elseif isequal(fun,@vqpe),  mytitle = ['VQPE: ',mytitle];
elseif isequal(fun,@uvqpe), mytitle = ['Unitary VQPE: ',mytitle];
end
plot_compare(t,lam,tol,E,mytitle,[0,Tmax],[1e-6,1]);
