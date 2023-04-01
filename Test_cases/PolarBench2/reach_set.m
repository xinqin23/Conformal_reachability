clear all
clc
close all

l0=[0.8; 0.5];
u0=[0.9; 0.6];

timestep=0.2;
T=35;  % horizon


dim = 2;

load('ControlBench1.mat')
load('s2s_Model_1epoch.mat')




% figure
% for j=1:1000
%     S2(:,1)=l0+rand(dim,1).*(u0-l0);
%     for i=1:T
%         S2(:,i+1)=NN(net, S2(:,i));
%     end
%     
%     plot(S2(1,:)-S2(4,:)-t_gap*S2(5,:)-D_default)
%     hold on
% end





figure
for j=1:1000
    S1(:,1)=l0+rand(dim,1).*(u0-l0);
    for i=1:T 
        init_a=[S1(:,i)];
        a_ego=pred(controller_nn, init_a);
        [~,in_out] =  ode45(@(t,x)dynamicsBench1(t,x,a_ego),[0 timestep],S1(:,i));
        S1(:,i+1)=in_out(end,:)';
        
    end
    
    plot(S1(1,:))
    hold on
end

function y = pred(net, x)
   
    len=length(net.weights)-1;
    for i=1:len
        x=poslin(net.weights{i}*x+net.biases{i});
    end
    y=net.weights{end}*x+net.biases{end};
    
end