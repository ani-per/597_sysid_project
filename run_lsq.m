%% Ani Perumalla. AERSP 597, Final Project.

%% Logistics

close all; clear all;

%%
% |startup| is a script I wrote that sets up some plot properties and directories for saving the plots.

startup;

%% Data formatting

airsim_data = load(fullfile(pwd, "data", "data_train_bulk.mat"));
t = squeeze(permute(airsim_data.t, [2, 3, 1]));
U = squeeze(permute(airsim_data.U, [2, 3, 1]));
Z = squeeze(permute(airsim_data.Z, [2, 3, 1]));
n_sim = size(t, 2);
dt = t(2, 1) - t(1, 1);

%% Finding region of constant longitudinal speed

i_start = zeros(n_sim, 1) - 1;
t_start = zeros(n_sim, 1) - 1;
v_x = zeros(n_sim, 1) - 1;
for n = 1:n_sim
    U_n = Z(3, :, n);
    for i = 1:length(U_n)
        pf = polyfit(t(i:end, n), U_n(:, i:end), 1);
        if (abs(pf(1)) < 1e-3)
            i_start(n) = i;
            t_start(n) = t(i);
            v_x(n) = mean(U_n(:, i:end));
            break 
        end
    end
end

%% Least squares estimation

input = [2];
U = U(input, :, :);
obs = [4, 6];
Z = Z(obs, :, :);

l = 5;
c = -10000;
m = zeros(n_sim, 1);
I = zeros(n_sim, 1);
rms_e = zeros(n_sim, 2);
for sim = 1:n_sim
    [a, rms_e(sim, :)] = lsq_mi(U(:, :, sim)', Z(:, :, sim)', dt, v_x(sim), l, c);
    m(sim) = 1/a(1);
    I(sim) = 1/a(2);
end

%% Plots

%%% Parameter Plots

which_plot = (m <= 2e4) & (m >= 0);
m_plot = m(which_plot);
I_plot = I(which_plot);

n_bins = 50;
mi_hist = figure("Position", fig_size);
tiles = tiledlayout(2, 1, "Padding", "none", "TileSpacing", "compact");
nexttile;
histogram(m_plot, n_bins); hold on;
xline(mean(m_plot), "-", sprintf("Mean: %0.0f $kg$", mean(m_plot)), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size);
xline(median(m_plot), "-", sprintf("Median: %0.0f $kg$", median(m_plot)), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xlabel("Mass ($m$) [$kg$]");
nexttile;
histogram(I_plot, n_bins); hold on;
xline(mean(I_plot), "-", sprintf("Mean: %0.0f $kg{\\cdot}m^2$", mean(I_plot)), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size);
xline(median(I_plot), "-", sprintf("Median: %0.0f $kg{\\cdot}m^2$", median(I_plot)), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xlabel("Moment of Inertia ($I_{zz}$) [$kg{\cdot}m^2$]");
title(tiles, ...
    [sprintf("Histogram of Parameters Estimated via Least-Squares over %i Simulations", n_sim); ...
    sprintf("$l = %i$ m, $c = %i$ $\\frac{N}{rad}$", l, c)], ...
    "Interpreter", "Latex", "FontSize", font_size);
export_fig(fullfile(root, "hist_m-I"), "-png", "-pdf");

%%% RMS Error Plots

rms_e_plot = rms_e(which_plot, :);

n_bins = 50;
mi_hist = figure("Position", fig_size);
tiles = tiledlayout(2, 1, "Padding", "none", "TileSpacing", "compact");
nexttile;
histogram(rms_e_plot(:, 1), n_bins); hold on;
xline(mean(rms_e_plot(:, 1)), "-", sprintf("Mean: %0.2f $\\frac{m}{s^2}$", mean(rms_e_plot(:, 1))), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size);
xline(median(rms_e_plot(:, 1)), "-", sprintf("Median: %0.2f $\\frac{m}{s^2}$", median(rms_e_plot(:, 1))), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xlabel("RMS Error in Rate of Change of Lateral Velocity ($\dot{V}$) [$\frac{m}{s^2}$]");
nexttile;
histogram(rms_e_plot(:, 2), n_bins); hold on;
xline(mean(rms_e_plot(:, 2)), "-", sprintf("Mean: %0.2f $\\frac{rad}{s^2}$", mean(rms_e_plot(:, 2))), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size);
xline(median(rms_e_plot(:, 2)), "-", sprintf("Median: %0.2f $\\frac{rad}{s^2}$", median(rms_e_plot(:, 2))), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xlabel("RMS Error in Rate of Change of Yaw Rate ($\dot{r}$) [$\frac{rad}{s^2}$]");
title(tiles, ...
    [sprintf("Histogram of RMS Error in Parameters Estimated via Least-Squares over %i Simulations", n_sim); ...
    sprintf("$l = %i$ m, $c = %i$ $\\frac{N}{rad}$", l, c)], ...
    "Interpreter", "Latex", "FontSize", font_size);
export_fig(fullfile(root, "hist_e_m-I"), "-png", "-pdf");

%% Least squares function to estimate mass and moment of inertia

function [a, rms_e] = lsq_mi(u, x, dt, v_x, l, c)
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
    rms_e = sqrt((sum(reshape(x_fd - H*a, nt - 1, 2), 1).^2)/(nt - 1));
end