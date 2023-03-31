clear all
clc
close all

addpath(genpath('----- NNV directory -----  \nnv'))
addpath(genpath('---- MOSEK directory-----  \Mosek'))
addpath(genpath('----Toolbox directory----  \src'))



load('s2s_Model.mat')
s2s_model = Net;
%%%%% be careful about the number of cells per parameter. It should be
%%%%% consistant with the STL specification you will introduce later,


% analysis_type='exact-star';
analysis_type='approx-star';

the_time = 46;

Center=[100;32.1;0;10.5;30.1;0];  %%% This is the center of the set of initial states $\mathcal{X}_0$. 
epsilon=[10;0.1;0;0.5;0.1;0];     %%% This is infinite norm radious of the set of initial states.
num_Core=6;

load('s2s_Data_trajectory_exact.mat')


%%%%%%%%%%%%%%%%%

spmd
  gpuDevice( 1 + mod( labindex - 1, gpuDeviceCount ) )
end

horizon = the_time + 3;

for i=1:horizon
    Input_Data{i} = gpuArray(Input_Data{i});
    Output_Data{i} = gpuArray(Output_Data{i});
end


%%%%%%%%%%%%%%%%


beta = 0.95;
delta = 1- (1-beta)/(6*horizon);
Conf_d =zeros(6, horizon);
tic
for i=1:horizon
    Conf_d(:,i) = Conf_apply(Input_Data{i}, Output_Data{i}, s2s_model, delta);
end
Conformal_time = toc;

tic
Star_sets = ReLUplex_Reachability_ss(Center, epsilon, s2s_model, analysis_type, num_Core, horizon, Conf_d);
Reachability_time = toc;

cd Results
save('ACC_approx_095_trajectory_exact', 'Conf_d','Conformal_time', 'Star_sets', 'Reachability_time' )
cd ..