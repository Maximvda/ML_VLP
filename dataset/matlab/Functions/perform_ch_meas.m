function [channel_data,swing,var_high,var_low] = perform_ch_meas(tx_id,rx_id,no_it,max_no_tx,max_no_rx,mx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    channel_data = zeros(max_no_tx,48,max_no_rx,no_it);
    swing = zeros(max_no_tx,max_no_rx,no_it);
    var_high = zeros(max_no_tx,max_no_rx,no_it);
    var_low = zeros(max_no_tx,max_no_rx,no_it);
    for h=1:no_it
        disp(['it=',num2str(h),'/',num2str(no_it)]);
        for i=1:length(tx_id)
            fwrite(mx,cast([1 ; tx_id(i)],'uint8'),'uint8');
%             disp('Channel measurement sent');


            [channel_data_raw_bytes_tmp,count,msg] = fread(mx,[98*4],'uint8');          
            if(isempty(channel_data_raw_bytes_tmp))
%                 disp(strcat('no data received for TX',num2str(tx_id(i))));
            elseif(length(channel_data_raw_bytes_tmp) == 98*4)
%                 disp(strcat('response from TX',num2str(tx_id(i))));
                for j=1:max_no_rx
                    tx_id_tmp = channel_data_raw_bytes_tmp((j-1)*98+1);
                    rx_id_tmp = channel_data_raw_bytes_tmp((j-1)*98+2)-20; % subtract 20 because we want id and not mac address
                    if(ismember(tx_id_tmp,tx_id) && ismember(rx_id_tmp,rx_id))
                        for k=1:48
                            channel_data(tx_id_tmp,k,rx_id_tmp,h) = channel_data_raw_bytes_tmp((j-1)*98+2*k+1) * 2^8 + channel_data_raw_bytes_tmp((j-1)*98+2*k+2);
                        end
                        [swing(tx_id_tmp,rx_id_tmp,h),var_high(tx_id_tmp,rx_id_tmp,h),var_low(tx_id_tmp,rx_id_tmp,h)] = getSwing(channel_data(tx_id_tmp,:,rx_id_tmp,h));
                    end
                end
            else
                disp(['error in length of read channel data. Length is: ' num2str(length(channel_data_raw_bytes_tmp))])
            end
        end
    end
end

