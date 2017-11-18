classdef Koncer_class
    properties
        X               % Huruf X ideal
        X1              % Huruf X translasi
        X2              % Huruf X tebal
        X3              % Huruf X kecil
        X4              % Huruf X rotasi
        O               % Huruf O ideal
        O1              % Huruf O translasi
        O2              % Huruf O tebal
        O3              % Huruf O kecil
        O4              % Huruf O rotasi
        W               % input volume size
        F               % conv filter size
        S               % conv stride (pergeseran)
        P               % conv padding (pinggiran)
        Segment         % jumlah segmen x
        x               % Filter Conv Layer x
        o               % Filter Conv Layer o
        s               % pool stride
        f               % pool filter size
        OX              % array of huruf
        name            % list huruf
        fclweight       % bobot fully connected layer
        Result          % hasil fully connected layer
    end
    methods
        
        % Convolution
        function V = ccn_conv(c,v,V)
            c.W = length(V);                    % input volume size
            Out = (c.W-c.F+2*c.P)/c.S + 1;      % size of output matrix
            V_temp = cell(Out,Out,c.Segment);   % cell containing sum of product
            if iscell(c.X) == 0                 % predetermined cell or matrix
                V_a = cell(length(V));
                for i=1:c.Segment
                    V_a(:,:,i)=num2cell(V);
                end
                V = V_a;
            end
            if c.P ~= 0                         % add padding
                V = padarray(cell2mat(V),[c.P,c.P],0,'both');
                V = num2cell(V);
            end
            for i=1:Out
                for j=1:Out
                    for k=1:c.Segment
                        temp = cell2mat(v(:,:,k)).*cell2mat(V(j:j+c.F-c.S,i:i+c.F-c.S,k));
                        V_temp(j,i,k) = num2cell(sum(sum(temp))/length(V));
                    end
                end
            end
            V = V_temp;                         % update value
        end
        
        % ReLU
        function V = ccn_relu(c,V)
            c.W = length(V);                  % input volume size
            for k=1:c.Segment
                for i=1:c.W
                    for j=1:c.W
                        if V{j,i,k} < 0
                            V(j,i,k) = {0};
                        end
                    end
                end
            end
        end
        
        % Pooling
        function V = ccn_pool(c,V)
            c.W = length(V);
            out = ceil(c.W/c.s);    % size of output matrix
            if mod(c.W,c.f)~=0
                V(length(V)+1,length(V)+1,:)={0};
                cell_kosong = cellfun('isempty',V);
                V(cell_kosong) = {0};
            end
            
            V_temp = cell(out,out,c.Segment);
            
            for i=1:out
                for j=1:out
                    for k=1:c.Segment
                        temp = cell2mat(V(c.s*(j-1)+1:c.s*j,c.s*(i-1)+1:c.s*i,k));
                        V_temp(j,i,k) = num2cell(max(max(temp)));
                    end
                end
            end
            V = V_temp;
        end
        
        % Matrix to Vector
        function V = ccn_mat2vec(c,V)
            V = cell2mat(V);
            [p1,p2,p3]=size(V);
            V = reshape(V,1,p1*p2*p3);
        end
        
        % Fully Connected Layer
        function Result = ccn_fcl(c,OX,fclweight)
            length_OX = length(OX);
            OX = cell2mat(OX);
            
            Result = zeros(1,length_OX);
            for i = 1:length_OX
                temp = OX(i,:).*fclweight;
                Result(i) = sum(temp);
            end
        end
        
    end
end