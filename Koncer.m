clear; clc;
c = Koncer_class;

%% Definisi Matrix & Parameter
c.X = csvread('Database/Koncer_X.csv');  % Huruf X ideal
c.X1 = csvread('Database/Koncer_X1.csv');% Huruf X translasi
c.X2 = csvread('Database/Koncer_X2.csv');% Huruf X tebal
c.X3 = csvread('Database/Koncer_X3.csv');% Huruf X kecil
c.X4 = csvread('Database/Koncer_X4.csv');% Huruf X rotasi
c.O = csvread('Database/Koncer_O.csv');  % Huruf O ideal
c.O1 = csvread('Database/Koncer_O1.csv');% Huruf O translasi
c.O2 = csvread('Database/Koncer_O2.csv');% Huruf O tebal
c.O3 = csvread('Database/Koncer_O3.csv');% Huruf O kecil
c.O4 = csvread('Database/Koncer_O4.csv');% Huruf O rotasi

c.F = 3;                        % conv filter size
c.S = 1;                        % conv stride (pergeseran)
c.P = 0;                        % conv padding (pinggiran)
c.s = 2;                        % pool stride
c.f = 2;                        % pool filter size
c.Segment = 3;                  % jumlah segmen x

c.x = cell(c.F,c.F,c.Segment);  % Filter Conv Layer
c.x(:,:,1) = num2cell(csvread('Database/Koncer_xa.csv'));
c.x(:,:,2) = num2cell(csvread('Database/Koncer_xb.csv'));
c.x(:,:,3) = num2cell(csvread('Database/Koncer_xc.csv'));

c.o = cell(c.F,c.F,c.Segment);  % Filter Conv Layer
c.o(:,:,1) = num2cell(csvread('Database/Koncer_oa.csv'));
c.o(:,:,2) = num2cell(csvread('Database/Koncer_ob.csv'));
c.o(:,:,3) = num2cell(csvread('Database/Koncer_oc.csv'));

%% Convolution - ReLU - Pooling
c.OX = {c.X; c.X1; c.X2; c.X3; c.X4; c.O; c.O1; c.O2; c.O3; c.O4};

for i = 1:length(c.OX)
    c.OX{i} = ccn_conv(c,c.x,c.OX{i});      % ubah c.x jadi c.o untuk melihat o
    c.OX{i} = ccn_relu(c,c.OX{i});
    c.OX{i} = ccn_pool(c,c.OX{i});
    c.OX{i} = ccn_relu(c,c.OX{i});
    c.OX{i} = ccn_pool(c,c.OX{i});
    c.OX{i} = ccn_mat2vec(c,c.OX{i});
end

%% Fully Connected Layer - Backpropagation
c.fclweight = [0.00224684643463300,-0.395987121857289,-0.583484135543836,0.0497646661257800,2.01060727375712,-1.04807613162537,1.40911020418549,-1.09640429980158,-0.449671058967820,-0.0232779509910170,0.102388639941691,-0.498847793497664]; % Dapet dari training Koncer_backprop.m
c.Result = ccn_fcl(c,c.OX,c.fclweight);
c.Result = c.Result - min(c.Result(:));
c.Result = c.Result ./ max(c.Result(:));

%% Print Result
c.name = {['X ideal'], ['X translasi'], ['X tebal'], ['X kecil'], ['X rotasi'], ['O ideal'], ['O translasi'], ['O tebal'], ['O kecil'], ['O rotasi']};
for i = 1:length(c.OX)
    fprintf('%s: %4.2f\n',c.name{i},c.Result(i));
end

clear i;