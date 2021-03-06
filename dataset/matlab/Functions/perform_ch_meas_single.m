function [channel_data,swing,var_high,var_low] = perform_ch_meas_single(tx_id,rx_id,no_it,mx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    no_tx = length(tx_id);

    channel_data = zeros(no_tx,48,no_it);
    swing = zeros(no_tx,no_it);
    var_high = zeros(no_tx,no_it);
    var_low = zeros(no_tx,no_it);
    for h=1:no_it
        disp(['it=',num2str(h),'/',num2str(no_it)]);
        for i=1:no_tx
            fwrite(mx,cast([4 ; tx_id(i) ; rx_id],'uint8'),'uint8');
%             disp('Channel measurement sent');


            [channel_data_raw_bytes_tmp,count,msg] = fread(mx,[98],'uint8');
            if(isempty(channel_data_raw_bytes_tmp))
%                 disp(strcat('no data received for TX',num2str(tx_id(i))));
            else
                disp(strcat('response from TX',num2str(tx_id(i))));
                tx_id_tmp = channel_data_raw_bytes_tmp(1);
                rx_id_tmp = channel_data_raw_bytes_tmp(2)-20; % subtract 20 because we want id and not mac address
                if(ismember(tx_id_tmp,tx_id) && rx_id_tmp == rx_id)
                    for k=1:48
                        channel_data(tx_id_tmp ==  tx_id,k,h) = channel_data_raw_bytes_tmp(2*k+1) * 2^8 + channel_data_raw_bytes_tmp(2*k+2);
                    end
                    [swing(tx_id_tmp ==  tx_id,h),var_high(tx_id_tmp ==  tx_id,h),var_low(tx_id_tmp ==  tx_id,h)] = getSwing(channel_data(tx_id_tmp == tx_id,:,h));
                end
            end
        end
    end
end

