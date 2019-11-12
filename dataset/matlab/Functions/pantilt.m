function  pantilt(tx_id,angle_polar,angle_azimuth,mx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    fwrite(mx,[cast([5;tx_id;angle_polar;angle_azimuth],'uint8')],'uint8');
end

