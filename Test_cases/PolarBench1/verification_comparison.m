clear all
clc
close all


addpath(genpath('----- NNV directory -----  \nnv'))
addpath(genpath('---- MOSEK directory-----  \Mosek'))
addpath(genpath('----Toolbox directory----  \src'))
addpath(genpath('----Toolbox directory----  \Test_cases\SherLock-ACC-problem_ss\Results'))


l0=[90;32;0;10;30;0];
u0=[110;32.2;0;11;30.2;0];


load('ACC_approx_090_trajectory_exact.mat')
H= length(Star_sets);
Lb_95 = zeros(6, H+1);
Ub_95 = zeros(6, H+1);

Lb_95(:,1) = l0;
Ub_95(:,1) = u0;

parfor i=1:H
    Box = Overall_Box( Star_sets{i}, eye(6), zeros(6,1));
    Lb_95(:,i+1) = Box(:,1);
    Ub_95(:,i+1) = Box(:,2);
end


load('ACC_approx_099_trajectory_exact.mat')
H= length(Star_sets);
Lb_97 = zeros(6, H+1);
Ub_97 = zeros(6, H+1);

Lb_97(:,1) = l0;
Ub_97(:,1) = u0;

parfor i=1:H
    Box = Overall_Box( Star_sets{i}, eye(6), zeros(6,1));
    Lb_97(:,i+1) = Box(:,1);
    Ub_97(:,i+1) = Box(:,2);
end



clearvars -except Lb_95 Lb_97 Ub_95  Ub_97 


%%%%% Download CORA from its github directory and place it in the directory mentioned below 
addpath(genpath('----Toolbox directory----\Test_cases\SherLock-ACC-problem_ss\CORA_2022'))
rmpath(genpath(' ------NNV directory------\nnv'))
 

R0 = interval([90; 32; 0; 10; 30; 0], [110; 32.2; 0; 11; 30.2; 0]);

params.tFinal = 4.9;
params.R0 = polyZonotope(R0);


% Reachability Settings ---------------------------------------------------
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.timeStep = 0.01;
options.taylorTerms = 4;
options.zonotopeOrder = 20;
options.alg = 'lin';
options.tensorOrder = 2;

% Parameters for NN evaluation --------------------------------------------
evParams = struct();
evParams.bound_approx = true;
evParams.polynomial_approx = "lin";


% System Dynamics ---------------------------------------------------------

% parameter
u_f = 0.0001;
a_lead = -2;
v_set = 30;
T_gap = 1.4;

% open-loop system
f = @(x, u) [x(2); x(3); -2 * x(3) + 2 * a_lead - u_f * x(2)^2; ...
             x(5); x(6); -2 * x(6) + 2 * u(1)   - u_f * x(5)^2];
sys = nonlinearSys(f);

% affine map x_ = C*x + k mapping state x to input of neural network x_
C = [0, 0, 0, 0, 0, 0; ...
     0, 0, 0, 0, 0, 0; ...
     0, 0, 0, 0, 1, 0; ...
     1, 0, 0,-1, 0, 0; ...
     0, 1, 0, 0,-1, 0];
k = [v_set; T_gap; 0; 0; 0];

load('Control.mat');

nn = neuralNetwork({
nnLinearLayer(controller_nn.weights{1}*C, controller_nn.weights{1}*k+controller_nn.biases{1});
nnReLULayer();
nnLinearLayer(controller_nn.weights{2}, controller_nn.biases{2});
nnReLULayer();
nnLinearLayer(controller_nn.weights{3}, controller_nn.biases{3});
nnReLULayer();
nnLinearLayer(controller_nn.weights{4}, controller_nn.biases{4});
});


% construct neural network controlled system
sys = neurNetContrSys(sys, nn, 0.1);


% Reachability Analysis -----------------------------------------------

R = reach(sys, params, options, evParams);

box on;
colors1='green';


t=0:0.1:4.9;
% plot results over time
for i=1:6
    figure(i)
    hold on
    plot(t,Lb_95(i,:), '--black')
    hold on
    plot(t,Lb_97(i,:), '-black')
    hold on
    plot(t,Ub_95(i,:), '--red')
    hold on
    plot(t,Ub_97(i,:), '-red')
    hold on
    plotOverTime(R,i,'FaceColor',colors1,'FaceAlpha',0.5 , 'EdgeColor' , 'none');
end


% print('-painters','-depsc','CasestudyEx2');
