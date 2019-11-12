function [channel_data_sel,swing_sel,var_high_sel,var_low_sel,indx_sel] = getBestChannelMeasurement_single_rx(channel_data,swing,var_high,var_low)
%GETBESTCHANNELMEASUREMENT Summary of this function goes here
%   Detailed explanation goes here
% dimensions: 
% swing: no_tx x no_rx x no_it
% channel_data: no_tx x 48 x no_rx x no_it
    for i=1:size(swing,1) % tx_id
        [swing_sel(i),indx_sel(i)] = max(swing(i,:));
        var_high_sel(i) = var_high(i,indx_sel(i));
        var_low_sel(i) = var_low(i,indx_sel(i));
        channel_data_sel(i,:) = channel_data(i,:,indx_sel(i));
    end
end

