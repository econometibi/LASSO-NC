function h=QRbandwidth(p,n,HSrule,alpha)
switch nargin
    case 0
        error('Minimum 2 inputs needed')
    case 1
        error('Minimum 2 inputs needed')
    case 2
        HSrule=1; %Hall-Sheather (1988) rule is default
        alpha=0.05;
    case 3
        alpha=0.05;
    case 4
end

x=norminv(p);
f=normpdf(x);

if HSrule==1
    h=n^(-1/3) * norminv(1-alpha/2)^(2/3) * ((1.5 * f^2)/(2*x^2 + 1))^(1/3); %Hall-Sheather (1988) rule
else
    h=n^-0.2 * ((4.5 * f^4)/(2 * x^2 + 1)^2)^0.2; %Bofinger rule
end

end