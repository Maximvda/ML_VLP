function [acro,mx] = startup(no_tx,max_no_rx,offset)
%STARTUP2 Summary of this function goes here
%   Detailed explanation goes here

    % These are the left corner positions of the regions where the CNC machines
    % can position the RXs
%     acro.offset_6x6(1,:) = [75 65];
%     acro.offset_6x6(2,:) = [75+1630 65];
%     acro.offset_6x6(3,:) = [75 65+1540];
%     acro.offset_6x6(4,:) = [75+1630 65+1540];
    
    
    acro.offset_6x6 = offset; 

    % Range of the CNC machines -> defines span where RXs can go
    acro.range_x = 1270;
    acro.range_y = 1290;
    acro.waittime_serial = 1;

    if(~isempty(instrfindall))
        fclose(instrfindall);
    end

    mx = init_socket(no_tx,max_no_rx);

    acro = init_acro(acro);
    acro = homeRXs(acro);
end

