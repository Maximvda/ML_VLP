% new_pos should be expressed in the dense deployment grid, in mm!!
function acro = moveToPos(acro,new_pos,tuning_offset,speed)
    
    speed_all = speed .* ones(1,4);

    for i=1:length(acro.serialObj)
        if(~any(new_pos(i,:) == inf))
            new_pos_abs(i,1) = new_pos(i,1)- acro.offset_6x6(i,1) - tuning_offset(i,1);
            new_pos_abs(i,2) = new_pos(i,2) - acro.offset_6x6(i,2) - tuning_offset(i,2);
            if(new_pos_abs(i,1) < 0 || new_pos_abs(i,1) > acro.range_x || new_pos_abs(i,2) < 0 || new_pos_abs(i,2) > acro.range_y) % check limits to avoid bumping HW limit switches
                disp(['cannot move RX' num2str(i) ' to this position']);
            else
                fprintf(acro.serialObj{i}, strcat('$J=X',num2str(new_pos_abs(i,1)),' Y',num2str(new_pos_abs(i,2)),' F',num2str(speed_all(i))));
                acro.pos(i,:) = new_pos(i,:);
            end
        end
    end
%     pause(1)
%     for i=1:length(acro.serialObj)
%         while(acro.serialObj{i}.BytesAvailable > 0)
%             fscanf(acro.serialObj{i});
%         end
%     end
end