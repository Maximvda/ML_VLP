function acro = init_acro(acro)

    % to get ports: udevadm info -a -n /dev/ttyUSB0 | grep '{serial}' |head -n1
    acro.serialObj{1} = serial('/dev/ttyUSB4');% serial('COM101');%Initialize serial communication
    acro.serialObj{2} = serial('/dev/ttyUSB1');%serial('COM102');%Initialize serial communication
    acro.serialObj{3} = serial('/dev/ttyUSB2');%serial('COM103');%Initialize serial communication
    acro.serialObj{4} = serial('/dev/ttyUSB3');%serial('COM104');%Initialize serial communication

    for i=1:length(acro.serialObj)
        disp('Opening serial ports');
        set(acro.serialObj{i},'BaudRate',115200);%Set Baudrate
        fopen(acro.serialObj{i});
    end

%     pause(0.5);
%     for i=1:length(acro.serialObj)
%         while(acro.serialObj{i}.BytesAvailable > 0)
%             disp(fscanf(acro.serialObj{i}));
%         end
%     end
end


