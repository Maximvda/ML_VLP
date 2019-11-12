function waitForIdle(acro,acro_id)
% DESCRIPTION: wait until all acro systems are idle

    pattern = '<(.*?)\|'; % to decode status in state information

    idle = ones(length(acro.serialObj),1);
    idle(acro_id) = 0; % we assume the other acro systems are already idle because position didn't change
    
    while (all(idle == 1) ~= 1)
        for i=1:length(acro.serialObj)
            if(idle(i) == 0)
                fprintf(acro.serialObj{i}, '?'); % request state information
                pause(acro.waittime_serial);
                
                response = '';
                while(acro.serialObj{i}.BytesAvailable > 0)
                    response = [response fscanf(acro.serialObj{i})];
                end
                [start,stop] = regexp(response,pattern);
                state = response(start+1:stop-1);
                disp([num2str(i) ':' state])
                if(strcmp(state,'Idle'))
                    idle(i) = 1;
                else
                    if(strcmp(response,''))
                        disp(['no response from serial port ' num2str(i) ' yet, trying again']);
                    elseif(isempty(state))
                        disp(['no state info in message from serial port ' num2str(i) ' yet, trying again']);   
                    end
                end
            end
        end
    end
end

