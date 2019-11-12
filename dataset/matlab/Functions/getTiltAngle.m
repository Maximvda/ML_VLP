function [angle_polar, angle_azimuth] = getTiltAngle(pos_tx, pos_rx)
%TILTTXTORX Summary of this function goes here
%   Detailed explanation goes here
    horz_dist = pdist([pos_tx(1:2); pos_rx(1) pos_rx(2)],'euclidean');
    vert_dist = pos_tx(3) - pos_rx(3);
    angle_tmp = atan2d(horz_dist,vert_dist);
    angle_polar_orig = wrapTo180(180-angle_tmp);
    angle_tmp2 = atan2d(pos_tx(2) - pos_rx(2),pos_tx(1) - pos_rx(1));
    angle_azimuth_orig = wrapTo180(angle_tmp2 + 180);
    
    if(angle_azimuth_orig < 0)
        angle_azimuth = -angle_azimuth_orig;
        angle_polar = angle_polar_orig - 90;
    else
        angle_azimuth = 180 - angle_azimuth_orig; % this is also equal to -angle_tmp2
%         angle_polar = angle_polar_orig;
        angle_polar = angle_tmp + 90;
    end
%     angle_azimuth = abs(wrapTo180(-angle_azimuth_orig));

end

