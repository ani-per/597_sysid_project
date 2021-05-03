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
n_sim = size(t_train, 2);

i_start_train = zeros(n_sim, 1) - 1;
t_start_train = zeros(n_sim, 1) - 1;
v_x_train = zeros(n_sim, 1) - 1;
for n = 1:n_sim
    U_n = Z_train(3, :, n);
    for i = 1:length(U_n)
        pf = polyfit(t_train(i:end, n), U_n(:, i:end), 1);
        if (abs(pf(1)) < 1e-3)
            i_start_train(n) = i;
            t_start_train(n) = t_train(i);
            v_x_train(n) = mean(U_n(:, i:end));
            break 
        end
    end
end

i_start_test = zeros(n_sim, 1) - 1;
t_start_test = zeros(n_sim, 1) - 1;
v_x_test = zeros(n_sim, 1) - 1;
for n = 1:n_sim
    U_n = Z_test(3, :, n);
    for i = 1:length(U_n)
        pf = polyfit(t_test(i:end, n), U_n(:, i:end), 1);
        if (abs(pf(1)) < 1e-3)
            i_start_test(n) = i;
            t_start_test(n) = t_test(i);
            v_x_test(n) = mean(U_n(:, i:end));
            break 
        end
    end
end

% input = [2];
% U_train = U_train(input, :);
% U_test = U_train(input, :);
% obs = [2, 4, 5, 6];
% Z_train = Z_train(obs, :, :);
% Z_test = Z_test(obs, :, :);