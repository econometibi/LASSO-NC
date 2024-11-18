function [output]=AIC_BIC(y,x,taus,bhat,BIC)
%AIC and BIC coded by Tibor Szendrei for Szendrei and Varga (2023). Based
%on Jiang,Wang, Bondell (2014).
[n,p]=size(x);
res=y-[ones(size(x,1),1),x]*bhat;
res_pos=res.*(res>0);
res_neg=-res.*(res<0);
QR_loss=sum(res_pos.*taus+res_neg.*(1-taus));
beta=round(bhat(2:end,:),4);
beta_count=sum(sum((beta~=0)));
if BIC==1
    output=sum(log(QR_loss))+(log(n)/(2*n))*beta_count;
else
    output=sum(log(QR_loss))+(1/n)*beta_count;
end
end