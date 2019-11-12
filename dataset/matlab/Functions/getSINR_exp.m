function [SINR] = getSINR_exp(alloc_matrix,channel_data_all,swings,var_high,var_low,desired_rx)
            l=1; % counter for interference
            k=1; % counter for desired signal
            swing_desired = 0;
            power_noise = 0;
            first = 1;
            sum_raw_interfering = 0;
            for i=1:size(alloc_matrix,1)
                tx_id  = alloc_matrix(i,1);
                rx_id  = alloc_matrix(i,2);

                if(rx_id == desired_rx)
                    swing_desired(k,1) = swings(tx_id,desired_rx);
                    sum_raw_desired =  squeeze(channel_data_all(tx_id,:,desired_rx));
                    if first == 1 % we assume that the most dominating noise comes from the strongest signal
                        power_noise = max([var_high(tx_id,desired_rx),var_low(tx_id,desired_rx)]);
                        first = 0;
                    end
                    k=k+1;
                elseif(rx_id ~= 0) 
                    sum_raw_interfering = sum_raw_interfering + ...
                        squeeze(channel_data_all(tx_id,:,desired_rx));
%                         circshift(squeeze(channel_data_all(tx_id,:,desired_rx)),randi(48));
                    l=l+1;
                end
            end
            power_interference = var(sum_raw_interfering);
            % compute SINR
    %         power_noise = 700;
    %         sum(swing_interfering)^2
            SINR = sum(swing_desired)^2/(power_interference+(power_noise+10));
%             SINR = var(sum_raw_desired)/(power_interference+(power_noise+10));

end

