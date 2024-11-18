function y = lp_fnm(A1, c1, A2, c2, b,  u, x1, x2)
% Solve a linear program by the interior point method:
% min{(c1*x1+c2*x2) | A1 * x1 + A2 * x2 = b and 0 < x1 < u and 0 < x2}
% An initial (possibly infeasible) solution has to be provided as x=[x1 x2]
% The starting point should satisfy 0<x1<u and 0<x2 but need not satisfy A*x=b.
% The matrices A1 and A2 may be specified in sparse form, see e.g.  triogram.m
%
% History:  This function is based on the Lustig, Marsden and Shanno(1992)
% description of an implementation of Mehrotra's (1990) predictor-corrector
% interior point method for bounded variable linear programming.  The code
% is loosely based on the Splus and fortran versions of the algorithm by
% Roger Koenker and described in Portnoy and Koenker (1997).  The first
% matlab version of the code was translated  by Paul Eilers in 1999
% from an Ox version written by Daniel Morillo and Roger Koenker.
% Modifications to handle sparsity and inequality constraints 1999-02.

% Roger Koenker:  Last revised April 20 2002.

% Set some constants
beta = 0.9995;
small = 1e-8;
max_it = 100000;
[m1,n1] = size(A1);
[m2,n2] = size(A2);

% Generate an initial point
s = u - x1;
y=(pinv(A1')*c1')'; %Adjusted by Tibor Szendrei
%y = (A1' \  c1')'; %ORIGINAL
r1 = c1 - y * A1;
r2 = c2 - y * A2;
z1 = r1 .* (r1 > 0);
w = z1 - r1;
z2 = ones(n2,1)';
gap = z1 * x1 + z2 * x2 + w * s;

% Start iterations
it = 0;
while gap > small && it < max_it
    it = it + 1;
    
    %   Compute affine step
    r1 = c1 - y * A1;
    r2 = c2 - y * A2;
    r3 = b - A1*x1 - A2*x2;
    q1 = 1 ./ (z1' ./ x1 + w' ./ s);
    q2 = x2 ./ z2';
    AQ1 = A1 * sparse(1:n1,1:n1,q1);
    AQ2 = A2 * sparse(1:n2,1:n2,q2);
    AQA = AQ1 * A1' + AQ2 * A2';
    rhs = r3 + AQ1 * r1' + AQ2 * r2';
    %ADDING TO TACKLE SINGULAR MATRIX (by Tibor Szendrei)
    %if isnan(det(AQA)) || det(AQA)==0
        dy = (pinv(AQA)*rhs)';
    %else
    %    dy = (AQA \ rhs)';
    %end
    %dy = (AQA \ rhs)'; %ORIGINAL
    dx1 = q1 .* (dy * A1 - r1)';
    dx2 = q2 .* (dy * A2 - r2)';
    ds = -dx1;
    dz1 = -z1 .* (1 + dx1 ./ x1)';
    dz2 = -z2 .* (1 + dx2 ./ x2)';
    dw = -w .* (1 + ds ./ s)';
    
    %   Compute maximum allowable step lengths
    fx1 = bound(x1,dx1);
    fx2 = bound(x2,dx2);
    fz1 = bound(z1,dz1);
    fz2 = bound(z2,dz2);
    fs = bound(s, ds);
    fw = bound(w, dw);
    fp = min(min(fx1, fs));
    fd = min(min(fw, fz1));
    fp = min(min(fx2),fp);
    fd = min(min(fz2),fd);
    fp = min(beta * fp, 1);
    fd = min(beta * fd, 1);
    
    %   If full affine step is feasible, take it. Otherwise modify it
    if min(fp, fd) < 1
        
        %     Update mu
        mu = z1 * x1 + z2 * x2 + w * s;
        g1 = (z1 + fd * dz1) * (x1 + fp * dx1);
        g2 = (z2 + fd * dz2) * (x2 + fp * dx2);
        g = g1 +g2 + (w + fd * dw) * (s + fp * ds);
        mu = mu * (g / mu) ^3 / ( 2* n1 + n2);
        
        %     Compute modified step
        dxdz1 = dx1 .* dz1';
        dxdz2 = dx2 .* dz2';
        dsdw = ds .* dw';
        xinv1 = 1 ./ x1;
        xinv2 = 1 ./ x2;
        sinv = 1 ./ s;
        xi1 = xinv1 .* dxdz1 - sinv .* dsdw - mu * (xinv1 - sinv);
        xi2 = xinv2 .* dxdz2  - mu * xinv2 ;
        rhs = rhs + A1 * ( q1 .* xi1) + A2 * (q2 .* xi2);
        %ADDING TO TACKLE SINGULAR MATRIX (by Tibor Szendrei)
        %if isnan(det(AQA)) || det(AQA)==0
            dy = (pinv(AQA)*rhs)';
        %else
        %    dy = (AQA \ rhs)';
        %end
        %dy = (AQA \ rhs)'; %ORIGINAL
        dx1 = q1 .* (A1' * dy' - r1' - xi1);
        dx2 = q2 .* (A2' * dy' - r2' - xi2);
        ds = -dx1;
        dz1 =  - z1 + xinv1' .* (mu - z1 .* dx1' - dxdz1');
        dz2 =  - z2 + xinv2' .* (mu - z2 .* dx2' - dxdz2');
        dw =  - w + sinv' .* (mu - w .* ds' - dsdw');
        
        %     Compute maximum allowable step lengths
        fx1 = bound(x1,dx1);
        fx2 = bound(x2,dx2);
        fz1 = bound(z1,dz1);
        fz2 = bound(z2,dz2);
        fs = bound(s, ds);
        fw = bound(w, dw);
        fp = min(min(fx1, fs));
        fd = min(min(fw, fz1));
        fp = min(min(fx2),fp);
        fd = min(min(fz2),fd);
        fp = min(beta * fp, 1);
        fd = min(beta * fd, 1);
        
    end
    
    %   Take the step
    x1 = x1 + fp * dx1;
    x2 = x2 + fp * dx2;
    z1 = z1 + fd * dz1;
    z2 = z2 + fd * dz2;
    s = s + fp * ds;
    y = y + fd * dy;
    w = w + fd * dw;
    
    gap = z1 * x1 + z2 * x2  + w * s;
    %disp(gap);
    if  it >= max_it
        warning('max_it exceeded: non-convergence')
    end
end
end