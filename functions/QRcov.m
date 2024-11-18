function cov=QRcov(x,y,coef,tau)

[n,p]=size(x);
h=QRbandwidth(tau,n);
if (tau+h>1)
    error('tau+h>1')
end
if (tau-h<0)
    error('tau-h<0')
end
u = y - x*coef;
h2=(norminv(tau+h) - norminv(tau-h)) * min(sqrt(var(u)),(quantile(u,0.75)-quantile(u,0.25))/1.34);
f=normpdf(u/h2)/h2;
[~,R]=qr(sqrt(f) .* x);
R=R(1:p,1:p);
fx=R\eye(p);
fx=fx*fx';
cov=tau*(1-tau).*(fx * x' * x * fx'); %calculation of covariance matrix
end