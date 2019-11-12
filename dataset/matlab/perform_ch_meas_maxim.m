close all;
clear all;

max_no_tx = 36;
max_no_rx = 4;
timestring = datestr(now,'yyyy-mm-dd_HH-MM-SS');

tx_id= [1:36];
no_tx = length(tx_id);

rx_id = [1 2 3 4];
no_it = 1;
%% set the offset
%default offset
offset(1,:) = [35 10];
offset(2,:) = [35+1630 10];
offset(3,:) = [35 10+1550];
offset(4,:) = [35+1630 10+1550];

if(~exist('acro','var'))
    disp('running startup script');
    [acro,mx] = startup(no_tx,max_no_rx,offset);
end

resolution = 10; % in mm
tuning_offset = zeros(4,2);
speed = 5000;

pos_x_mm = 0:resolution:1200;
pos_y_mm = 0:resolution:1200;

inverty = 0;
number_of_meas = 5;

acro_id = rx_id;
for i=1:length(pos_x_mm)
    disp(['pos_x=',num2str(i),'/',num2str(length(pos_x_mm))]);
    for j=1:length(pos_y_mm)
        if(inverty)
            y_inv = length(pos_y_mm)-j+1;
        else
            y_inv = j;
        end
        disp(['pos_y=',num2str(y_inv),'/',num2str(length(pos_y_mm))]);

        % move the RX
        pos_rx_mm_tmp = acro.pos;

        % position RXs in the middle (to reduce waiting time)
        pos_rx_mm_tmp = acro.offset_6x6 + 625*ones(4,2);
        pos_rx_mm_tmp(acro_id,:) = acro.offset_6x6(acro_id,:) + [pos_x_mm(i) pos_y_mm(y_inv)];
        pos_changed = pos_rx_mm_tmp ~= acro.pos;
        acro_id_check = find(or(pos_changed(:,1),pos_changed(:,2)));

        % move the acro systems
        acro = moveToPos(acro,pos_rx_mm_tmp,tuning_offset,speed);

        waitForIdle(acro,acro_id_check);

        % channel_data: tx_id x sample_number x rx_id x it_id x pos_x x pos_y
        % swing: tx_id x rx_id x it_id x pos_x x pos_y
            %         [channel_data(:,:,:,:,i,j),swing_tmp,var_high(:,:,:,i,j),var_low(:,:,:,i,j)] = perform_ch_meas(tx_id,rx_id,no_it,max_no_rx,mx);
        for meas_i=1:number_of_meas
            [channel_data_tmp,swing_tmp,var_high_tmp,var_low_tmp] = perform_ch_meas(tx_id,rx_id,no_it,max_no_tx,max_no_rx,mx);
            channel_data(:,:,rx_id,:,i,y_inv,meas_i) = channel_data_tmp(:,:,rx_id,:);
            var_high(:,rx_id,:,i,y_inv,meas_i) = var_high_tmp(:,rx_id,:);
            var_low(:,rx_id,:,i,y_inv,meas_i) = var_low_tmp(:,rx_id,:);
            squeeze(swing_tmp(tx_id,rx_id,:))
            swing(:,rx_id,:,i,y_inv,meas_i) = swing_tmp(:,rx_id,:);
        end
        
    save(strcat('testbed_data_',timestring,'.mat'));
    end
    if(inverty)
        inverty = 0
    else
        inverty = 1
    end
end

disp('Close sockets');
if(~isempty(mx))
    fclose(mx);
    delete(mx);
end
