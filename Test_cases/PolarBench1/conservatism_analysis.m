clear all
clc
close all

addpath(genpath('----- NNV directory -----  \nnv'))
addpath(genpath('---- MOSEK directory-----  \Mosek'))
addpath(genpath('----Toolbox directory----  \src'))
addpath(genpath('----Toolbox directory----  \Test_cases\SherLock-ACC-problem_ss\Results'))

load('Bench1_approx_095_trajectory_exact.mat')

H= length(Star_sets);


l0=[0.8; 0.5];
u0=[0.9; 0.6];

dim = 2;

Lb = zeros(dim, H+1);
Ub = zeros(dim, H+1);

Lb(:,1) = l0;
Ub(:,1) = u0;

parfor i=1:H
    Box = Overall_Box( Star_sets{i}, eye(dim), zeros(dim,1));
    Lb(:,i+1) = Box(:,1);
    Ub(:,i+1) = Box(:,2);
end




timestep=0.2;

load('ControlBench1.mat')

num_traj = 10;
num_plot = 10;



S = cell(1,num_traj);


accepted = 0;
parfor j=1:num_traj
    S{j}(:,1)=l0+rand(6,1).*(u0-l0);
    for i=1:H
        
        init_a=[S{i}(:,j)];
        a_ego=cntr(nn, init_a);
        [~,in_out] =  ode45(@(t,x)dynamicsBench1(t,x,a_ego),[0 timestep],S{i}(:,j)');
        in_out=in_out(end,:)';
        S{j}(:,i+1) = in_out(end,:)';
        
    end

    logical = min([(S{j}-Ub<=0)  ;  (S{j}-Lb>=0)  ]);
    
    accepted = accepted + min(logical);

end

beta_emp = accepted / num_traj;

clearvars -except Lb  Ub  S times num_traj  num_plot beta_emp



t=0:0.1:4.9;
% plot results over time
for i=1:6
    figure(i)
    hold on
    plot(t,Lb(i,:))
    hold on
    plot(t,Ub(i,:))
    hold on
    plotOverTime(R,i,'FaceColor',colors1,'FaceAlpha',0.5 , 'EdgeColor' , 'none');
    
    for j=1:num_plot
        k =floor(rand*num_traj)+1;
        plot(t, S{k}(i,:))
        hold on
    end
end


% clearvars -except Lb  Ub  S  beta_emp  R

function y = pred(net, x)
   
    len=length(net.weights)-1;
    for i=1:len
        x=poslin(net.weights{i}*x+net.biases{i});
    end
    y=net.weights{end}*x+net.biases{end};
    
end