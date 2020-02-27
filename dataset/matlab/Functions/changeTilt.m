function changeTilt(sockets, range_pan, range_tilt)
    for socket = sockets
        %ran_pan = (rand-0.5)*range_pan
        %ran_tilt = (rand-0.5)*range_tilt
        ran_pan = normrnd(0,range_pan/3);
        ran_tilt = normrnd(0,range_tilt/3);
        message = [num2str(ran_pan, '%08.3f') num2str(ran_tilt, '%08.3f')];
        write(socket, message);
    end
end