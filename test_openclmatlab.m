ocl = opencl();
ocl.initialize(1,1);
ocl.addfile('cl/simple_add.cl');
ocl.build();

% Test buffer:
n = int32(10);
x = clbuffer('ro', 'single', n);
y = clbuffer('ro', 'single', n);
z = clbuffer('rw', 'single', n);
p = clbuffer('wo', 'single', n);

x.set([1:10]);
y.set([21:30]);
z.set([0.1:0.1:1]*3);
p.set([3:12]);

% Validate clbuffer
x.get()
y.get()
z.get()
p.get()

% Validate clkernel
addkernel = clkernel('add', [n, 0,0], [n,0,0]);
addkernel(x,y,z,n);

values = z.get();
values

clear x y z p;

% Test clobject
x = clobject(single(1:10));
y = clobject(single(11:20));
z = clobject(zeros(1,10, 'single'));

addkernel(x,y,z, uint32(10));
values = z.get();

x.get()
y.get()
values


