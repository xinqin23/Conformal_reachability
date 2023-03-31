clear all
clc
close all



lb = [0.8; 0.5];
ub = [0.9;0.6];

load('ControlBench1.mat')
nn.weights=controller_nn.weights;
nn.biases =controller_nn.biases;
normalization=0;
timestep=0.2;
num_traj= 40000 ;
horizon = 30; % [0,7], 0.2
[theInput, theOutput, maxmin] = bench1_Datagenerator_ss(lb, ub, nn, timestep, normalization, num_traj, horizon);

Input_Data = theInput;
Output_Data = theOutput;
save('s2s_Data_trajectory_exact.mat','Input_Data', 'Output_Data', 'maxmin');

clear all
