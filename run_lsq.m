%% Ani Perumalla. AERSP 597, Final Project.

%% Logistics

close all; clear all;

%%
% |startup| is a script I wrote that sets up some plot properties and directories for saving the plots.

startup;

%% 

train_data = load(fullfile(pwd, "data", "data_train_bulk.mat"));
test_data = load(fullfile(pwd, "data", "data_test_bulk.mat"));
t_train = squeeze(permute(train_data.t, [2, 3, 1]));
U_train = squeeze(permute(train_data.U, [2, 3, 1]));
Z_train = squeeze(permute(train_data.Z, [2, 3, 1]));
t_test = squeeze(permute(train_data.t, [2, 3, 1]));
U_test = squeeze(permute(test_data.U, [2, 3, 1]));
Z_test = squeeze(permute(test_data.Z, [2, 3, 1]));