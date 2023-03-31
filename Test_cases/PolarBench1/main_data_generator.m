clear all
clc
close all

Center=[100;32.1;0;10.5;30.1;0]; 
epsilon=[10;0.1;0;0.5;0.1;0];

lb = (Center-epsilon);
ub = (Center+epsilon);

load('ControlBench1.mat')
nn.weights=controller_nn.weights;
nn.biases =controller_nn.biases;
normalization=0;
timestep=0.1;
num_traj= 40000 ;
horizon = 50;
[theInput, theOutput, maxmin] = ACC_nln_Datagenerator_ss(lb, ub, nn, timestep, normalization, num_traj, horizon);

Input_Data = theInput;
Output_Data = theOutput;
save('s2s_Data_trajectory_exact.mat','Input_Data', 'Output_Data', 'maxmin');

clear all
