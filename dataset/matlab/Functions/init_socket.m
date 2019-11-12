function mx = init_socket(no_tx,max_no_rx)
    buffer_out = 8*max_no_rx; % byte length % we send 8 bytes per RX
    buffer_in = 98*max_no_rx; % data for no_rx RXs
    mx = udp('192.168.7.127',1112,'LocalPort',1113,'OutputBufferSize', buffer_out, 'OutputDatagramPacketSize', buffer_out, ...
        'InputBufferSize', buffer_in, 'InputDatagramPacketSize', buffer_in); % mx
    set(mx, 'Timeout',1);
    if(~isempty(mx))
        fopen(mx);
    end
end

