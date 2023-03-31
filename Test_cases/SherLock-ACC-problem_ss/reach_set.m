clear all
clc
close all

l0=[90;32;0;10;30;0];
u0=[110;32.2;0;11;30.2;0];

timestep=0.1;
T=50;
V_set=30;
t_gap=1.4;
E=[ 0  0  0  0  1  0;...
    1  0  0 -1  0  0;...
    0  1  0  0 -1  0];
D_default=10;

load('Control.mat')
load('s2s_Model.mat')




figure
for j=1:1000
    S2(:,1)=l0+rand(6,1).*(u0-l0);
    for i=1:T
        S2(:,i+1)=NN(Net, S2(:,i));
    end
    
    plot(S2(1,:)-S2(4,:)-t_gap*S2(5,:)-D_default)
    hold on
end





figure
for j=1:1000
    S1(:,1)=l0+rand(6,1).*(u0-l0);
    for i=1:T
        
        init_a=[V_set; t_gap; E*S1(:,i)];
        a_ego=pred(controller_nn, init_a);
        [~,in_out] =  ode45(@(t,x)dynamicsACC(t,x,a_ego),[0 timestep],S1(:,i));
        S1(:,i+1)=in_out(end,:)';
        
    end
    
    plot(S1(1,:)-S1(4,:)-t_gap*S1(5,:)-D_default)
    hold on
end

function y = pred(net, x)
   
    len=length(net.weights)-1;
    for i=1:len
        x=poslin(net.weights{i}*x+net.biases{i});
    end
    y=net.weights{end}*x+net.biases{end};
    
end