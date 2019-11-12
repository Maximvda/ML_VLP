close all;
clear all;

max_no_tx = 36;
max_no_rx = 4;
timestring = datestr(now,'yyyy-mm-dd_HH-MM-SS');

tx_id= [1:36];
no_tx = length(tx_id);

rx_id = [1 2 3 4];
no_it = 1;
deployment = 1;
tilt = 0;

tx_tilt_id = [8 11 26 29 15];
tx_tilt_pos_mm = [750 750 ; 2250 750 ; 750 2250 ; 2250 2250 ; 1500 1500];
tx_tilt_pos_mm(:,3) = 1750;


%% set the offset
%default offset
offset(1,:) = [35 10];
offset(2,:) = [35+1630 10];
offset(3,:) = [35 10+1550];
offset(4,:) = [35+1630 10+1550];

%% set the deployment id
if(deployment == 1) % default setting
    pos_x_limits = [1 13];
    pos_y_limits = [1 13];
elseif(deployment == 2) % cnc machines translated over x axis to cover whole region
% cnc machine is translated for 30cm to the right
    offset = offset + 300*[ones(4,1) zeros(4,1)];
    pos_x_limits = [10 13];
    pos_y_limits = [1 13];
elseif(deployment == 3) % cnc machines translated over y axis to cover whole region
% cnc machine is translated for 40cm to above
    offset = offset + 300*[zeros(4,1) ones(4,1)];
    pos_x_limits = [1 13];
    pos_y_limits = [10 13];
elseif(deployment == 4) % cnc machines translated over y axis to cover whole region
% cnc machine is translated for 40cm to above
    offset = offset + 300*[ones(4,1) ones(4,1)];
    pos_x_limits = [10 13];
    pos_y_limits = [10 13];
end

if(tilt)
    no_seq = length(rx_id);
    rx_id_seq = rx_id;
else
    no_seq = 1;
end


if(~exist('acro','var'))
    disp('running startup script');
    [acro,mx] = startup(no_tx,max_no_rx,offset);
end

resolution = 10; % in mm
tuning_offset = zeros(4,2);
speed = 5000;

pos_x_mm = 0:resolution:1200;
pos_y_mm = 0:resolution:1200;


for r=1:no_seq
    if(tilt)
        % move one RX after the other
        rx_id = rx_id_seq(r);
        acro_id = rx_id_seq(r);
    else
        % move all RXs together
        acro_id = rx_id;
    end
    for i=pos_x_limits(1):pos_x_limits(2)
        disp(['pos_x=',num2str(i),'/',num2str(length(pos_x_mm))]);
        for j=pos_y_limits(1):pos_y_limits(2)
            disp(['pos_y=',num2str(j),'/',num2str(length(pos_y_mm))]);

            % move the RX
            pos_rx_mm_tmp = acro.pos;

            % position RXs in the middle (to reduce waiting time)
            pos_rx_mm_tmp = acro.offset_6x6 + 625*ones(4,2);
            pos_rx_mm_tmp(acro_id,:) = acro.offset_6x6(acro_id,:) + [pos_x_mm(i) pos_y_mm(j)];
            pos_changed = pos_rx_mm_tmp ~= acro.pos;
            acro_id_check = find(or(pos_changed(:,1),pos_changed(:,2)));

            % move the acro systems
            acro = moveToPos(acro,pos_rx_mm_tmp,tuning_offset,speed);

            if(tilt)
            % tilt the 4 dynamic TXs towards the RX
                for k=1:length(tx_tilt_id)
                    [angle_polar(rx_id,i,j,k), angle_azimuth(rx_id,i,j,k)] = getTiltAngle(tx_tilt_pos_mm(k,:)/1000, [pos_rx_mm_tmp(acro_id,:) 0]/1000);
                    if(ismember(tx_tilt_id(k),tx_id))
                        pantilt(tx_tilt_id(k),round(angle_polar(rx_id,i,j,k)),round(angle_azimuth(rx_id,i,j,k)),mx);
                    end
                end
            else
                for k=1:length(tx_tilt_id)
                    angle_polar(rx_id,i,j,k) = 90;
                    angle_azimuth(rx_id,i,j,k) = 90;
                    pantilt(tx_tilt_id(k),round(angle_polar(rx_id,i,j,k)),round(angle_azimuth(rx_id,i,j,k)),mx);
                end
            end


            waitForIdle(acro,acro_id_check);

            % channel_data: tx_id x sample_number x rx_id x it_id x pos_x x pos_y
            % swing: tx_id x rx_id x it_id x pos_x x pos_y
                %         [channel_data(:,:,:,:,i,j),swing_tmp,var_high(:,:,:,i,j),var_low(:,:,:,i,j)] = perform_ch_meas(tx_id,rx_id,no_it,max_no_rx,mx);

            [channel_data_tmp,swing_tmp,var_high_tmp,var_low_tmp] = perform_ch_meas(tx_id,rx_id,no_it,max_no_tx,max_no_rx,mx);
            channel_data(:,:,rx_id,:,i,j) = channel_data_tmp(:,:,rx_id,:);
            var_high(:,rx_id,:,i,j) = var_high_tmp(:,rx_id,:);
            var_low(:,rx_id,:,i,j) = var_low_tmp(:,rx_id,:);
            squeeze(swing_tmp(tx_id,rx_id,:))
            swing(:,rx_id,:,i,j) = swing_tmp(:,rx_id,:);
        end
        save(strcat('intermediate_result_D',num2str(deployment),'_',timestring,'.mat'));
%         notifyEmail(strcat('rx_id=',num2str(rx_id),'| i=',num2str(i),'| j=',num2str(j)));
    end
end

disp('Close sockets');
if(~isempty(mx))
    fclose(mx);
    delete(mx);
end
