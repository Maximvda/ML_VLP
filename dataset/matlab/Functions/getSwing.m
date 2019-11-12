function [swing,var_high,var_low] = getSwing(data)
%GETSWING Summary of this function goes here
%   Detailed explanation goes here
    avg = mean(data);

    k=1;l=1;
    data_low = 0;
    data_high = 0;
    for i=1:length(data)
        if(data(i) >= avg)
            data_high(k) = data(i);
            k=k+1;
        else
            data_low(l) = data(i);
            l=l+1;
        end
    end
    
    swing = mean(data_high) - mean(data_low);
    var_high = var(data_high);
    var_low = var(data_low);
end

