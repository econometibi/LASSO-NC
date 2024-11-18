% This is the code for "Revisiting vulnerable growth in the Euro Area:
% Identifying the role of financial conditions in the distribution" by
% Szendrei and Varga (2023). Please cite the work when using the code.

clear all
close all
clc
%% Parameters

excl_list=[1];                  %Exclusion from shrinkage (Constant automatically added, 1=first column of X)
h=1;                            %Forecast horizon: h={1,4}
taus=[0.1:0.1:0.9];             %Estimated quantiles
BICparam=2;                     %BIC or AIC: AIC=1 BIC=2
tmax=150;                       %Maximum variation allowed
increm=0.5;                     %Fineness of lambda grid
AdaLASSO=1;                     %Adaptive LASSO choice
LASSORUN=1;                     %First time LASSO run=1
LASSO_or_not=1;                 %Use LASSO betas or normal QR betas for selecting BIC
forcstart=50;                   %Start OOS forecast
fontsize=15;

%% Data preamble
mainpath=cd;
funcs=mainpath+"/functions/";

addpath(funcs);

[data,label]=xlsread("LASSO_GaR_DB.xlsx","PreCovDB"); %Change between "PreCovDB" and "FullDB"
BigY=data(1:end,3:6);
BigY_label=label(1,2:5);
BigX=data(1:end,7:end);
BigX_label=label(1,6:end);

mat=[BigY(:,5-h),BigX];
mat(any(isnan(mat), 2), :) = []; %drop NaN rows

%Final set of data
x=mat(:,2:end);
y=mat(:,1);

labelused=['Constant', BigX_label];

clearvars BigX_label BigX BigY BigY_label data mat

%% Variable selection starts here
tvect=(0:increm:tmax);

if LASSORUN==1
    bhat_vec_LASSO=nan(size(x,2)+1,size(taus,2),size(tvect,2));
    bhat_vec=bhat_vec_LASSO;
    BIC_vec_LASSO=zeros(size(tvect,2),2);
    BIC_vec=BIC_vec_LASSO;
    
    f = waitbar(0, 'Starting');
    for i=1:size(tvect,2)
        %Run LASSO
        bhat=round(VaribSelectNC(y,x,taus,tvect(i),excl_list,AdaLASSO),4);
        bhat_vec_LASSO(:,:,i)=bhat;
        
        %Calculate AIC/BIC
        BIC_vec_LASSO(i,1)=AIC_BIC(y,x,taus,bhat,0);
        BIC_vec_LASSO(i,2)=AIC_BIC(y,x,taus,bhat,1);
        
        %Refit model with selected variables
        Position=(bhat==0);
        Position_vec=reshape((bhat==0),size(Position,1)*size(Position,2),1)';
        bhat=round(ConstrainedFitNC(y,x,taus,Position_vec),4);
        bhat_vec(:,:,i)=bhat;
        
        %Calculate AIC/BIC
        BIC_vec(i,1)=AIC_BIC(y,x,taus,bhat,0);
        BIC_vec(i,2)=AIC_BIC(y,x,taus,bhat,1);
        
        %Progress bar update
        waitbar(i/size(tvect,2), f, sprintf('Progress: %d %%', floor(i/size(tvect,2)*100)));
    end
    close(f)
    
    filename = mainpath+"\output\Coeff_h"+h+".mat";
    save(filename,'bhat_vec_LASSO','BIC_vec_LASSO','bhat_vec','BIC_vec','labelused');
else
    if h==1
        load('output/Coeff_h1.mat')
    else
        load('output/Coeff_h4.mat')
    end
end

%Optimal Lambda selection
start=1;

if LASSO_or_not==1
    AICBIC_vec=BIC_vec_LASSO;
else
    AICBIC_vec=BIC_vec;
end
[min_val, min_index] = min(AICBIC_vec(start:end,BICparam));
bhat=bhat_vec(:,:,min_index+start-1);

%Fitted values
yf=[ones(size(x,1),1),x]*bhat;

%Display results
disp(" ");
disp("Final Beta estimates:")
[labelused',num2cell(bhat)]
file=mainpath+"\output\OptCoeff_h"+h+".xlsx";
writecell(ans,file);

%% Figures

%Variables across lambda
[~,medloc]=min(abs(taus-0.5));
subplotsloc=[1,medloc,length(taus)];
fig=figure('DefaultAxesFontSize',fontsize);
set(gcf, 'Position', get(0, 'Screensize'),...
    'DefaultAxesColorOrder',[1 0 0; 0 1 0; 0 0 1; 1 0 1; 0 1 1],...
    'DefaultAxesLineStyleOrder',{'-','--',':','-.'});
tcl=tiledlayout(1,3);
for s=1:length(subplotsloc)
    coeff_mat=squeeze(bhat_vec_LASSO(length(excl_list)+2:end,subplotsloc(s),:))';
    tauused=taus(subplotsloc(s))*100;
    titleused=tauused+"^{th} Quantile";
    
    %subplot(1,3,s);
    nexttile(tcl);
    plot(tvect,coeff_mat);
    
    [~, min_index] = min(AICBIC_vec(start:end,1));
    xline(tvect(min_index),'-','AIC');
    [~, min_index] = min(AICBIC_vec(start:end,2));
    xline(tvect(min_index),'-','BIC');
    %legend(labelused(3:end),'Location','southwest');
    
    xlabel('Maximum variation');
    ylabel('LASSO Coefficient');
    title(titleused);
end
hL=legend(labelused(length(excl_list)+2:end));
hL.Layout.Tile='East';
file=mainpath+"\output\VaribSel_h"+h+".jpg";
saveas(fig,file)

%% In Sample Comparison with CISS-GDP model
%Normal QR
Normal_QR=[];
for i=1:length(taus)
    temp_bhat=rq_fnm([ones(size(y,1),1),x(:,1:length(excl_list)+1)], y, taus(i)); %Function taken from R. Koenker website.
    Normal_QR=[Normal_QR,temp_bhat];
end
yfqr=[ones(size(x,1),1),x(:,1:length(excl_list)+1)]*Normal_QR;

%NC CISS
Position=ones(size(bhat));
Position(1:length(excl_list)+2,:)=0;
Position_vec_CISS=reshape(Position,size(Position,1)*size(Position,2),1)';
NC_CISS=round(ConstrainedFitNC(y,x,taus,Position_vec_CISS),4);

yfnc=[ones(size(x,1),1),x]*NC_CISS;

%Comparison  in sample
[~,medloc]=min(abs(taus-0.5));
tauloc=[1,medloc,length(taus)];
resmat=nan(3,3);
for t=1:length(tauloc)
    tauuse=taus(tauloc(t));
    testmat=[yf(:,tauloc(t)),yfnc(:,tauloc(t)),yfqr(:,tauloc(t))];
    for i=1:size(testmat,2)
        u=y-testmat(:,i);
        v1=sum(0.5*(abs(u)+(2*tauuse-1).*u));
        n=length(u);
        if t==1
            beta=bhat(:,tauloc(t));
            beta_count=sum(sum((beta~=0)));
        else
            beta_count=3;
        end
        resmat(t,i)=log(v1)+(log(n)/(2*n))*beta_count; %BIC
    end
end
resmat=["Opt mod","NC CISS","QR CISS";num2cell(resmat)];

%% OOS Comparison with CISS-GDP model
Position=(bhat==0);
Position_vec_Opt=reshape((bhat==0),size(Position,1)*size(Position,2),1)';

OOSforc_Opt=nan(length(y),length(taus));
OOSforc_NC=OOSforc_Opt;
OOSforc_QR=OOSforc_Opt;

totnum=length(y)-forcstart+1;
num=0;
f = waitbar(0, 'Starting');
for forc=forcstart:length(y)
    num=num+1;
    yfit=y(1:forc-1);
    xfit=x(1:forc-1,:);
    xtest=x(forc,:);
    
    %Normal QR
    Normal_QR=[];
    for i=1:length(taus)
        temp_bhat=rq_fnm([ones(size(yfit,1),1),xfit(:,1:length(excl_list)+1)], yfit, taus(i)); %Function taken from R. Koenker website.
        Normal_QR=[Normal_QR,temp_bhat];
    end
    OOSforc_QR(forc,:)=[1,xtest(:,1:length(excl_list)+1)]*Normal_QR;
    
    %NC CISS
    NC_CISS=round(ConstrainedFitNC(yfit,xfit,taus,Position_vec_CISS),4);
    OOSforc_NC(forc,:)=[1,xtest]*NC_CISS;
    
    %Opt Model
    Opt_mod=round(ConstrainedFitNC(yfit,xfit,taus,Position_vec_Opt),4);
    OOSforc_Opt(forc,:)=[1,xtest]*Opt_mod;
    
    %Progress bar update
    waitbar(num/totnum, f, sprintf('Progress: %d %%', floor(num/totnum*100)));
end
close(f)

yf=y(forcstart:length(y));

resmat=[resmat;" "," "," "];
for i=2:4
    wQSres=[];
    for t=1:3
        if t==1
            yfcomp=OOSforc_Opt;
        elseif t==2
            yfcomp=OOSforc_NC;
        else
            yfcomp=OOSforc_QR;
        end
        output = wQS(yf,yfcomp(forcstart:length(y),:), taus, i);
        wQSres=[wQSres,round(mean(output),4)];
    end
    resmat=[resmat;wQSres];
end
labforres=[" ";"ISq10";"ISq50";"ISq90";" ";"QS (centre)";"QS (left tail)";"QS (right tail)"];
[labforres,resmat]

file=mainpath+"\output\OOSres_h"+h+".xlsx";
writematrix(ans,file);
