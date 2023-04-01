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


load('ControlBench1.mat')

num_traj = 30;
num_plot = 30;



S = cell(1,num_traj);
timestep = 0.2

accepted = 0;
parfor j=1:num_traj
%     S{j}(:,1)=l0+rand(6,1).*(u0-l0);
    initial=l0 + rand(dim,1).*(u0-l0);
    S{j}(:,1) = initial;
    for i=1:H
        init_a=[S{j}(:,i)];
        a_ego=pred(controller_nn, init_a);
        [~,in_out] =  ode45(@(t,x)dynamicsBench1(t,x,a_ego),[0 timestep],S{j}(:,i)');
        S{j}(:,i+1) = in_out(end,:)';

%         init_a = [V_set; t_gap; E*S{j}(:,i)];
%         a_ego = pred(controller_nn, init_a);
%         [~,in_out] =  ode45(@(t,x)dynamicsACC(t,x,a_ego),[0 timestep],S{j}(:,i));
%         S{j}(:,i+1) = in_out(end,:)';
        
    end

    logical = min([(S{j}-Ub<=0)  ;  (S{j}-Lb>=0)  ]);
    
    accepted = accepted + min(logical);

end

clearvars -except Lb  Ub  S times num_traj  num_plot beta_emp dim

figure(3)
t=0:0.2:7;
t=0:2:70;

for j=1:num_plot
    k =floor(rand*num_traj)+1;
    plot3(t, S{k}(1,:),S{k}(2,:), '-green')
    hold on
end

% plot results over time

hold on
plot3(t, Lb(1,:),Lb(2,:),'-black')
hold on
plot3(t, Ub(1,:),Ub(2,:),'-red')
hold on

xlabel('$t$','FontSize',16,'Interpreter','latex');
ylabel('$x_1$','FontSize',16,'Interpreter','latex');
zlabel('$x_2$','FontSize',16,'Interpreter','latex');
box on


% plotOverTime(R,i,'FaceColor',colors1,'FaceAlpha',0.5 , 'EdgeColor' , 'none');
saveas(figure(3),sprintf('bound2dim.png'))
% print(figure(3),'bound2dim','-depsc')

t=0:0.2:7;
t=0:2:70;

% plot results over time
for i=1:dim
    figure(i)
    hold on
%     plot(t,Lb(i,:),'black.')
    plot(t,Lb(i,:),'-black')

    hold on
%     plot(t,Ub(i,:),'r.')
    plot(t,Ub(i,:),'-red')

    hold on
%     plotOverTime(R,i,'FaceColor',colors1,'FaceAlpha',0.5 , 'EdgeColor' , 'none');
    
    for j=1:num_plot
        k =floor(rand*num_traj)+1;
%         plot(t, S{k}(i,:),'g.')
        plot(t, S{k}(i,:),'-green')
        
hold on
    end
    
    xlabel('$t$','FontSize',16,'Interpreter','latex');
    if i == 1
    ylabel('$x_1$','FontSize',16,'Interpreter','latex')
    else
         ylabel('$x_2$','FontSize',16,'Interpreter','latex')
    end
    box on

%     set(gca,'Xtick',0:2:70,'FontSize',16)
    saveas(figure(i),sprintf('bound%d.png', i))
    print(figure(i),sprintf('bound%d', i),'-depsc')

end

 
% clearvars -except Lb  Ub  S  beta_emp  R

function y = pred(net, x)
   
    len=length(net.weights)-1;
    for i=1:len
        x=poslin(net.weights{i}*x+net.biases{i});
    end
    y=net.weights{end}*x+net.biases{end};
    
end