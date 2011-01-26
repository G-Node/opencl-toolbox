ocl = opencl();
ocl.initialize(1,1);
ocl.addfile('matlab_kernels_float.cl');
ocl.build();

hostA = single([0.1:0.1:3]);
hostB = single(hostA+0.5);
hostC = zeros(size(hostA), 'single');

nElems = uint32(numel(hostA));

devA = clobject(hostA);
devB = clobject(hostB);
devC = clobject(hostC);

add = clkernel('single_add', [128, 0,0], [128,0,0]);
sub = clkernel('single_minus', [128, 0,0], [128,0,0]);
div = clkernel('single_divide', [128, 0,0], [128,0,0]);
mul = clkernel('single_times', [128, 0,0], [128,0,0]);
fexp = clkernel('single_exponential', [128,0,0], [128,0,0]);

add(devC, devA, devB, nElems); refC = hostA + hostB;
max(abs(devC.get() - refC))

sub(devC, devA, devB, nElems); refC = hostA - hostB;
max(abs(devC.get() - refC))

div(devC, devA, devB, nElems); refC = hostA ./ hostB;
max(abs(devC.get() - refC))

mul(devC, devA, devB, nElems); refC = hostA .* hostB;
max(abs(devC.get() - refC))

fexp(devC, devA, nElems); refC = exp(hostA);
max(abs(devC.get() - refC))


