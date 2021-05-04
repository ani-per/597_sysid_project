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
dt = t_train(2, 1) - t_train(1, 1);

i_start_train = zeros(n_sim, 1) - 1;
t_start_train = zeros(n_sim, 1) - 1;
v_x_train = zeros(n_sim, 1) - 1;
i_start_test = zeros(n_sim, 1) - 1;
t_start_test = zeros(n_sim, 1) - 1;
v_x_test = zeros(n_sim, 1) - 1;
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

input = [2];
U_train = U_train(input, :, :);
U_test = U_test(input, :, :);
obs = [4, 6];
Z_train = Z_train(obs, :, :);
Z_test = Z_test(obs, :, :);

l = 5;
c = -10000;
m_train = zeros(n_sim, 1);
I_train = zeros(n_sim, 1);
for sim = 1:n_sim
    a = lsq_mi(U_train(:, :, sim)', Z_train(:, :, sim)', dt, v_x_train(sim), l, c);
    m_train(sim) = 1/a(1);
    I_train(sim) = 1/a(2);
end

n_bins = 20;
mi_hist = figure("Position", fig_size);
tiles = tiledlayout(2, 1, "Padding", "none", "TileSpacing", "compact");
nexttile;
histogram(m_train, n_bins);
xlabel("Mass ($m$) [$kg$]");
nexttile;
histogram(I_train, n_bins);
xlabel("Mass ($m$) [$kg$]");
title(tiles, ...
    [sprintf("Histogram of Parameters Estimated via Least-Squares over %i Simulations", n_sim); ...
    sprintf("$l = %i$ m, $c = %i$ $\\frac{N}{rad}$", l, c)], ...
    "Interpreter", "Latex", "FontSize", font_size);
export_fig(fullfile(root, "hist_m-I"), "-pdf", "-png");

function [a] = lsq_mi(u, x, dt, v_x, l, c)
    arguments
        u (:, 1) double
        x (:, 2) double
        dt (1, 1) double
        v_x (1, 1) double
        l (1, 1) double = 5
        c (1, 1) double = -10000
    end

    assert(height(u) == height(x));
    
    nt = height(u);
    v_y = x(1:(end - 1), 1);
    r = x(1:(end - 1), 2);
    delta = u(1:(end - 1));
    
    x_fd = reshape(diff(x), [], 1)/dt + [v_x*r; zeros(size(r))];
    H = [(2*c/v_x)*v_y - c*delta, zeros(nt - 1, 1); zeros(nt - 1, 1), ((l^2*c)/(2*v_x))*r - (l*c/2)*delta];
    a = (H'*H)\H'*x_fd;
end