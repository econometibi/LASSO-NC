function [output,output_full] = wQS(TrueY, ForecastQuants, Quantiles, weighting)
%Gneiting and Ranjan (2011) type weights used in Szendrei and Varga (2023).


T=size(TrueY,1);
qs=max(size(Quantiles));

output=NaN(T,1); %preallocate for results
QS=output; %preallocate for QS scores

for t=1:T
    for q=1:qs
        resid=TrueY(t)-ForecastQuants(t,q); %Residual of the forecasted quantile
        ticklossw=Quantiles(q)-(TrueY(t)<=ForecastQuants(t,q)); %Tick loss weight assigned to residual
        QS(t,q)=ticklossw*resid; %gives unweighted QS for each quantile
        %Note that this is the QR tickloss used for forecast evaluation
    end
end

if weighting==1 %equal weights
    QStemp=QS;
    
elseif weighting==2 %Center gets more weight
    QStemp=NaN(T,q);
    for q=1:qs
        QStemp(:,q)=QS(:,q)*(Quantiles(q)*(1-Quantiles(q))); %Multiply with q(1-q)
    end
    
    
elseif weighting==3 %Left tail gets more weight
    QStemp=NaN(T,q);
    for q=1:qs
        QStemp(:,q)=QS(:,q)*((1-Quantiles(q))^2); %Multiple with (1-q)^2
    end
    
elseif weighting==4 %Rightail gets more weight
    QStemp=NaN(T,q);
    for q=1:qs
        QStemp(:,q)=QS(:,q)*(Quantiles(q)); %Multiple with (1-q)^2
    end
end

output=mean(QStemp,2); %Take the average
output_full=QS;

end