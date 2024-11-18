function [output]=VaribSelectNC(y,x,taus,tstar,excl_list,adaptive)
%Coded by Tibor Szendrei following Bondell, Reich, Wang (2010) and Jiang,
%Wang, Bondell (2014). Function used in Szendrei and Varga (2023) for 
%variable selection.
if nargin<6
    adaptive=0;
end
if nargin<5
    excl_list=[1];
else
    excl_list=[1,excl_list+1];
end

[n,p]=size(x);
m=max(size(taus));
pp=p+1; %for constant
Y=repmat(y,m,1); %repeat y for each quantiles
xtemp=x;

%SCALING
x=zeros(n,p);
shifts=min(xtemp);
x=xtemp-repmat(shifts,n,1);
scalings=max(x);
x=x./repmat(scalings,n,1);

%Save the scaling and shift for later: will need to do the inverse xform on
%the betas (Fine to do since QR is nonlinear transformation equivariant)

%MAKE X MATRIX FOR JOINT ESTIMATION
x=[ones(n,1),x];
D=eye(m);
X=kron(D,x);

%Run normal QR to get weights if ADALASSO
if adaptive==1
    Normal_QR=[];
    for i=1:length(taus)
        temp_bhat=rq_fnm(x, y, taus(i)); %Function taken from R. Koenker website.
        Normal_QR=[Normal_QR,temp_bhat];
    end
    ada_w=reshape(Normal_QR,1,[]);
    ada_w=1./abs(ada_w);
else
    ada_w=ones(1,(length(taus)*(size(x,2)+1)));
end

X2=[X,-X]; %This is needed for LP. Keeping all parameters to be larger than zero makes the constraints easier to code/handle.
%Converting X2 to sparse matrix helps in computation speed

K=m*pp;

%Adding vector of 0's for the differences
zeromat=zeros(size(X2,1),(m-1)*pp);
X2=[X2,zeromat,-zeromat];

%RESTRICTIONS
%Restrictions will be Rb>=r
%We want constant to be larger than the sum of negative differences. R1 is
%for positive constants (this should be positive), R2 is for negative coefficients

%Difference definition
defin=[-1,zeros(1,p),1,zeros(1,p),zeros(1,pp*(m-2)),...
    1,zeros(1,p),-1,zeros(1,p),...
    zeros(1,pp*(m-2)),...
    -1,zeros(1,p),...
    zeros(1,pp*(m-2)),...
    1,zeros(1,p),...
    zeros(1,pp*(m-2))];

constr1=defin;
for i=1:(pp*(m-1)-1)
    temp=[zeros(1,i),defin(1,1:(end-i))];
    constr1=[constr1;temp];
end
diffdef=[constr1;-constr1];

%NC constraint
R1temp=[1,zeros(1,p)];
R1=kron(eye(m-1),R1temp);

R2temp=ones(1,pp);
R2=kron(eye(m-1),R2temp);

NC=[zeros(size(R1,1),2*K),R1,-R2];

%LASSO constraint
VaribSel=[ones(1,K).*ada_w,ones(1,K).*ada_w,zeromat(1,:),zeromat(1,:)];

for i=1:length(excl_list)
    VaribSel(:,excl_list(i):pp:end)=0; %Variables we don't want to shrink (Constant always unshrunk)
end

%All constraints together
R=[-VaribSel;diffdef;eye(2*K+2*(K-pp));NC]; %Add identity matrix to the top
%Converting R to sparse matrix helps in computation speed

%r
r=[-tstar;zeros(size(diffdef,1),1);zeros(2*K+2*(K-pp)+(m-1),1)];

%b
x1=[];
for i=1:m
    temp=taus(i)*ones(n,1);
    x1=[x1;temp];
end
b=X2'*(1-x1);

%Other parameters needed for QR
u=ones(size(x1,1),1);
x2=ones(size(R,1),1);

%ESTIMATION
%Uses interior point method
coeff1=-lp_fnm(X2',-Y',R',-r',b,u,x1,x2); %Functions from Roger Koenker's rqic codefile

%Calculate coefficients using solution of interior metod
coeff=coeff1(1:K)-coeff1((K+1):2*K);
bhat_temp=reshape(coeff,[],m);

%Calculate transformation matrix
transform=[1,-shifts./scalings];
transform=[transform;zeros(p,1),diag(1./scalings)];

%Transform betas back
bhat=transform*bhat_temp; %final betas
output=bhat;
end