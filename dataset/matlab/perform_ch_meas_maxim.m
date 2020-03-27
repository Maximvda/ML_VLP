close all;
clear all;

%%Read in most recent file and load corresponding variables
%d=dir('*.mat');
%dd = zeros(1,length(d));
%for k = 1:length(d)
%    dd(k) = datenum(d(k).date);
%end
%[tmp i]=max(dd);
%d(i).name
%load(d(i).name,'i','j');
%i_start = i;
%j_start = j;

height = 176;%192; %in cm
max_no_tx = 36;
max_no_rx = 4;

tx_id= [1:36];
no_tx = length(tx_id);

rx_id = [1 2 3 4];
no_it = 3;
%% set the offset
%default offset
offset(1,:) = [0 0];
offset(2,:) = [1610 0];
offset(3,:) = [0 1550];
offset(4,:) = [1610 1550];

if(~exist('acro','var'))
    disp('running startup script');
    [acro,mx] = startup(no_tx,max_no_rx,offset);
end

resolution = 10; % in mm
tuning_offset = zeros(4,2);
speed = 5000;

pos_x_mm = 0:resolution:1200;
pos_y_mm = 20:resolution:1220;

inverty = 0;

acro_id = rx_id;

%% set initial position
% init_pos = [pos_x_mm(47) pos_y_mm(1)];
% pos_rx_mm_tmp = acro.offset_6x6 + 625*ones(4,2);
% pos_rx_mm_tmp(acro_id,:) = acro.offset_6x6(acro_id,:) + init_pos;
% pos_changed = pos_rx_mm_tmp ~= acro.pos;
% acro_id_check = find(or(pos_changed(:,1),pos_changed(:,2)));
% move the acro systems
%acro = moveToPos(acro,pos_rx_mm_tmp,tuning_offset,speed);
%waitForIdle(acro,acro_id_check)

%% loop over positions
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

        %waitForIdle(acro,acro_id_check);
        pause(resolution/10);       %1second delay for each cm

        % channel_data: tx_id x sample_number x rx_id x it_id x pos_x x pos_y
        % swing: tx_id x rx_id x it_id x pos_x x pos_y
            %         [channel_data(:,:,:,:,i,j),swing_tmp,var_high(:,:,:,i,j),var_low(:,:,:,i,j)] = perform_ch_meas(tx_id,rx_id,no_it,max_no_rx,mx);

        [channel_data_tmp,swing_tmp,var_high_tmp,var_low_tmp] = perform_ch_meas(tx_id,rx_id,no_it,max_no_tx,max_no_rx,mx);
        channel_data(:,:,rx_id,:,y_inv) = channel_data_tmp(:,:,rx_id,:);
        var_high(:,rx_id,:,y_inv) = var_high_tmp(:,rx_id,:);
        var_low(:,rx_id,:,y_inv) = var_low_tmp(:,rx_id,:);
%         squeeze(swing_tmp(tx_id,rx_id,:))
        swing(:,rx_id,:,y_inv) = swing_tmp(:,rx_id,:);
    end
    filestring = strcat(num2str(resolution),'_',num2str(height),'_row_',num2str(i));
    save(strcat('separate_files/testbed_data_',filestring,'.mat'));
    if(inverty)
        inverty = 0;
    else
        inverty = 1;
    end
end

disp('Close sockets');
if(~isempty(mx))
    fclose(mx);
    delete(mx);
end
