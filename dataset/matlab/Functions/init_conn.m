function s = init_conn(no_rx)
    port = 5022;
    for rx = no_rx
        ip = ['rx' num2str(rx,'%i')];
        s(rx) = tcpclient(ip,port);
    end
end
