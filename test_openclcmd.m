% Get platform info
platforms = openclcmd();

platforms
platforms(1).devices


% Initialize to platform 1, device 1:
openclcmd('initialize', 0,0);

% Load cl/VectorAdd.cl and build:
openclcmd('addfile', 'cl/simple_add.cl');
openclcmd('build');


% Create required buffers:
buffA = openclcmd('create_buffer', 'ro', uint32(4*9));
buffB = openclcmd('create_buffer', 'ro', uint32(4*9));
buffC = openclcmd('create_buffer', 'rw', uint32(4*9));

openclcmd('set_buffer', 0, buffA, single([1,2,3,4,5,6,7,8,9]));
openclcmd('set_buffer', 0, buffB, single([1,2,3,4,5,6,7,8,9]));

rA = openclcmd('get_buffer', 0, buffA, 9, 'single')
rB = openclcmd('get_buffer', 0, buffB, 9, 'single')

kid = openclcmd('create_kernel', uint32([9,0,0]), uint32([9,0,0]), 'add');
openclcmd('set_kernel_args', kid, 0, buffA, [], 0);
openclcmd('set_kernel_args', kid, 1, buffB, [], 0);
openclcmd('set_kernel_args', kid, 2, buffC, [], 0);
openclcmd('set_kernel_args', kid, 3, -1, int32(9), 0);
openclcmd('execute_kernel', 0, kid);
openclcmd('wait_queue', 0);
rC = openclcmd('get_buffer', 0, buffC, 9, 'single')

