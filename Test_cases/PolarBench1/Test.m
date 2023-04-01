clear all
clc
close all

addpath(genpath('----- NNV directory -----  \nnv'))
addpath(genpath('---- MOSEK directory-----  \Mosek'))
addpath(genpath('----Toolbox directory----  \src'))
addpath('src/TNN')
addpath('src/Reachability')
addpath('src/conf_bound')




load('s2s_Model_1epoch.mat')
s2s_model = net;
%%%%% be careful about the number of cells per parameter. It should be
%%%%% consistant with the STL specification you will introduce later,


% analysis_type='exact-star';
analysis_type='approx-star';


Center= [0.85;0.55];  %%% This is the center of the set of initial states $\mathcal{X}_0$. 
epsilon=[0.05; 0.05];     %%% This is infinite norm radious of the set of initial states.
num_Core=1;

load('s2s_Data_trajectory_exact.mat')


%%%%%%%%%%%%%%%%%

% spmd
%   gpuDevice( 1 + mod( labindex - 1, gpuDeviceCount ) )
% end

horizon = 35; 

% for i=1:horizon
%     Input_Data{i} = gpuArray(Input_Data{i});
%     Output_Data{i} = gpuArray(Output_Data{i});
% end


%%%%%%%%%%%%%%%%


beta = 0.95;
delta = 1- (1-nthroot(beta, horizon))/2
Conf_d =zeros(2, horizon);
tic
for i=1:horizon
    Conf_d(:,i) = Conf_apply(Input_Data{i}, Output_Data{i}, s2s_model, delta);
end
Conformal_time = toc;

tic
Star_sets = ReLUplex_Reachability_ss(Center, epsilon, s2s_model, analysis_type, num_Core, horizon, Conf_d);
Reachability_time = toc;

% cd Results
save('ACC_approx_095_trajectory_exact', 'Conf_d','Conformal_time', 'Star_sets', 'Reachability_time' )
% cd ..
