%% Ani Perumalla. AERSP 597, Final Project.

%% Logistics

close all; clear all;

%%
% |startup| is a script I wrote that sets up some plot properties and directories for saving the plots.

startup;
warning("off", "all");

%% Data formatting

airsim_data = load(fullfile(pwd, "data", "data_bulk_cc.mat"));
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
        pf = polyfit(t(i:end, n)', U_n(:, i:end), 1);
        if (abs(pf(1)) < 5e-3)
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

m = 2500;
I = 5445;
a = 1.441;
b = 2.977 - 1.441;
cf = zeros(n_sim, 1);
cr = zeros(n_sim, 1);
rms_e = zeros(n_sim, 2);
for sim = 1:n_sim
    [q, rms_e(sim, :)] = lsq_cc(U(:, :, sim)', Z(:, :, sim)', dt, v_x(sim), m, I, a, b);
    cf(sim) = q(1);
    cr(sim) = q(2);
end

%% Plots

%%% Parameter Plots

which_plot = (v_x > 0);
cf_plot = cf(which_plot);
cr_plot = cr(which_plot);

n_bins = 50;
mi_hist = figure("Position", fig_size);
tiles = tiledlayout(2, 1, "Padding", "none", "TileSpacing", "compact");
nexttile;
histogram(cf_plot, n_bins); hold on;
xline(mean(cf_plot), "-", sprintf("Mean: %0.0f $\\frac{N}{rad}$", mean(cf_plot)), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xline(median(cf_plot), "-", sprintf("Median: %0.0f $\\frac{N}{rad}$", median(cf_plot)), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "right");
xlabel("Front Tire Cornering Stiffness ($C_{{\alpha}f}$) [$\frac{N}{rad}$]");
nexttile;
histogram(cr_plot, n_bins); hold on;
xline(mean(cr_plot), "-", sprintf("Mean: %0.0f $\\frac{N}{rad}$", mean(cr_plot)), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xline(median(cr_plot), "-", sprintf("Median: %0.0f $\\frac{N}{rad}$", median(cr_plot)), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "right");
xlabel("Rear Tire Cornering Stiffness ($C_{{\alpha}r}$) [$\frac{N}{rad}$]");
title(tiles, ...
    [sprintf("Histogram of Parameters Estimated via Least-Squares over %i Simulations", sum(which_plot)); ...
    sprintf("$m = %0.0f$ kg, $I = %0.0f$ $kg{\\cdot}m^2$, $a = %0.3f$ m, $b = %0.3f$ m", m, I, a, b)], ...
    "Interpreter", "Latex", "FontSize", font_size);
export_fig(fullfile(root, "hist_cc"), "-png", "-pdf");

%%% RMS Error Plots

rms_e_plot = rms_e(which_plot, :);

n_bins = 50;
mi_e_hist = figure("Position", fig_size);
tiles = tiledlayout(2, 1, "Padding", "none", "TileSpacing", "compact");
nexttile;
histogram(rms_e_plot(:, 1), n_bins); hold on;
xline(mean(rms_e_plot(:, 1)), "-", sprintf("Mean: %0.2f $\\frac{m}{s^2}$", mean(rms_e_plot(:, 1))), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "right");
xline(median(rms_e_plot(:, 1)), "-", sprintf("Median: %0.2f $\\frac{m}{s^2}$", median(rms_e_plot(:, 1))), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xlabel("RMS Error in Rate of Change of Lateral Velocity ($\dot{V}$) [$\frac{m}{s^2}$]");
nexttile;
histogram(rms_e_plot(:, 2), n_bins); hold on;
xline(mean(rms_e_plot(:, 2)), "-", sprintf("Mean: %0.2f $\\frac{rad}{s^2}$", mean(rms_e_plot(:, 2))), ...
    "Color", colors(2, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "right");
xline(median(rms_e_plot(:, 2)), "-", sprintf("Median: %0.2f $\\frac{rad}{s^2}$", median(rms_e_plot(:, 2))), ...
    "Color", colors(3, :), "LineWidth", 2*line_width, "Interpreter", "latex", "FontSize", font_size, ...
    "LabelHorizontalAlignment", "left");
xlabel("RMS Error in Rate of Change of Yaw Rate ($\dot{r}$) [$\frac{rad}{s^2}$]");
title(tiles, ...
    [sprintf("Histogram of RMS Error in Parameters Estimated via Least-Squares over %i Simulations", sum(which_plot)); ...
    sprintf("$m = %0.0f$ kg, $I = %0.0f$ $kg{\\cdot}m^2$, $a = %0.3f$ m, $b = %0.3f$ m", m, I, a, b)], ...
    "Interpreter", "Latex", "FontSize", font_size);
export_fig(fullfile(root, "hist_e_cc"), "-png", "-pdf");

%% Least squares function to estimate tire cornering stiffness coefficients

function [q, rms_e] = lsq_cc(u, x, dt, v_x, m, I, a, b)
    arguments
        u (:, 1) double % Input vector
        x (:, 2) double % (Observed) state vector
        dt (1, 1) double % Timestep in s
        v_x (1, 1) double % Longitudinal velocity in m/s
        m (1, 1) double % Car mass in kg
        I (1, 1) double % Car yaw moment of inertia in kg*m^2
        a (1, 1) double % Distance from front axle to center of gravity in m
        b (1, 1) double % Distance from rear axle to center of gravity in m
    end

    assert(height(u) == height(x));
    
    nt = height(u);
    v_y = x(1:(end - 1), 1);
    r = x(1:(end - 1), 2);
    delta = u(1:(end - 1));
    
    x_fd = reshape(diff(x), [], 1)/dt + [v_x*r; zeros(size(r))];
    H = [ ...
        (1/(m*v_x))*v_y + (a/(m*v_x))*r - (1/m)*delta, (1/(m*v_x))*v_y - (b/(m*v_x))*r; ...
        (a/(I*v_x))*v_y + ((a^2)/(I*v_x))*r - (a/I)*delta, -(b/(I*v_x))*v_y + ((b^2)/(I*v_x))*r ...
        ];
    q = (H'*H)\H'*x_fd;
    rms_e = sqrt((sum(reshape(x_fd - H*q, nt - 1, 2), 1).^2)/(nt - 1));
end