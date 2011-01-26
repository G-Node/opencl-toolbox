function test_operations()
    % Setup workspace
    clear all;
    ocl = opencl();
    ocl.initialize(1,1);
    ocl.addfile('cl/matlab_kernels_float.cl');
    ocl.build();

    A = 1:10;
    B = 2:11;

    a = clfloat(1:10);
    b = clfloat(2:11);
        
    c = a+b; test_eq(A+B, c.get(), 'A+B');    
    c = a+3; test_eq(A+3, c.get(), 'A+3');    
    c = 3+a; test_eq(3+A, c.get(), '3+A');    
    
    c = a-b; test_eq(A-B, c.get(), 'A-B');    
    c = b-a; test_eq(B-A, c.get(), 'B-A');
    c = a-3; test_eq(A-3, c.get(), 'A-3');
    c = 3-a; test_eq(3-A, c.get(), '3-A');
    
    tol = 1e-5;
    c = a.*b; test_near(A.*B, c.get(), tol, 'A*B');    
    c = b.*a; test_near(B.*A, c.get(), tol, 'B*A');
    c = a.*3; test_near(A.*3, c.get(), tol, 'A*3');
    c = 3.*a; test_near(3.*A, c.get(), tol, '3*A');
        
    c = exp(a); test_near(exp(A), c.get(), 1e-2, 'exp(A)');    
    
end

function test_eq(expected, actual, name)
    fprintf(1, '%s : ', name);
    if all(expected == actual), 
        fprintf(1, '[SUCCESS]\n');
    else        
        fprintf(1, '[FAILED!]\n');
        fprintf(1, 'Got      '); disp(actual);
        fprintf(1, 'Expected '); disp(expected);
    end
end
    
function test_near(expected, actual, tol, name)
    fprintf(1, '%s : ', name);
    
    if (all(abs(expected - actual) < tol)), 
        fprintf(1, '[SUCCESS]\n');
    else        
        fprintf(1, '[FAILED!]\n');
        fprintf(1, 'Got      '); disp(actual);
        fprintf(1, 'Expected '); disp(expected); 
        fprintf(1, 'Max. Diff = %0.8f\n', max(abs(expected - actual)));
    end
end
