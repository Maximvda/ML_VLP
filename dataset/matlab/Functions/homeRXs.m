function acro = homeRXs(acro)
    for i=1:length(acro.serialObj)
%         while(acro.serialObj{i}.BytesAvailable > 0)
%                 disp(fscanf(acro.serialObj{i}));
%         end
        disp('Homing');
        fprintf(acro.serialObj{i}, '$X');
        pause(acro.waittime_serial);
%         while(acro.serialObj{i}.BytesAvailable > 0)
%             disp(fscanf(acro.serialObj{i}));
%         end
        fprintf(acro.serialObj{i}, '$H');
        pause(acro.waittime_serial);
%         while(acro.serialObj{i}.BytesAvailable > 0)
%             disp(fscanf(acro.serialObj{i}));
%         end
    end
   
    waitForIdle(acro,1:4);
    
    
    for i=1:length(acro.serialObj)
        fprintf(acro.serialObj{i}, 'G10 P0 L20 X0 Y0 Z0');
%         pause(1)
%         while(acro.serialObj{i}.BytesAvailable > 0)
%             fscanf(acro.serialObj{i});
%         end
    end
    
    acro.pos = acro.offset_6x6;
end