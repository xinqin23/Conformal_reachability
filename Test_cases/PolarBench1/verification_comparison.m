clear all
clc
close all


addpath(genpath('----- NNV directory -----  \nnv'))
addpath(genpath('---- MOSEK directory-----  \Mosek'))
addpath(genpath('----Toolbox directory----  \src'))
addpath(genpath('----Toolbox directory----  \Test_cases\SherLock-ACC-problem_ss\Results'))

addpath('..\nnv')


l0=[0.8; 0.5];
u0=[0.9; 0.6];
dim = 2;

load('Bench1_approx_095_trajectory_exact.mat')
H= length(Star_sets);
Lb_97 = zeros(dim, H+1);
Ub_97 = zeros(dim, H+1);

Lb_97(:,1) = l0;
Ub_97(:,1) = u0;

parfor i=1:H
    Box = Overall_Box( Star_sets{i}, eye(dim), zeros(dim,1));
    Lb_97(:,i+1) = Box(:,1);
    Ub_97(:,i+1) = Box(:,2);
end



clearvars -except Lb_95 Lb_97 Ub_95  Ub_97  dim


t=0:0.2:7;
% plot results over time
% for i=1:dim
%     figure(i)
%     hold on
% %     plot(t,Lb_95(i,:), '--black')
% %     hold on
%     plot(t,Lb_97(i,:), '-black')
%     hold on
% %     plot(t,Ub_95(i,:), '--red')
% %     hold on
%     plot(t,Ub_97(i,:), '-red')
%     hold on
% %     plotOverTime(R,i,'FaceColor',colors1,'FaceAlpha',0.5 , 'EdgeColor' , 'none');
% end


figure()
hold on
%     plot(t,Lb_95(i,:), '--black')
%     hold on
plot(Lb_97(1,:),Lb_97(2,:), '-black')
hold on
%     plot(t,Ub_95(i,:), '--red')
%     hold on
plot(Ub_97(1,:),Ub_97(2,:), '-red')
hold on
%     plotOverTime(R,i,'FaceColor',colors1,'FaceAlpha',0.5 , 'EdgeColor' , 'none');



% print('-painters','-depsc','CasestudyEx2');
