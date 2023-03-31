function [theInput, theOutput, maxmin] = bench1_Datagenerator_ss(lb, ub, nn, timestep, normalization, num_traj , horizon) 


n=2;


Nend=n;
N0=Nend;
Input = cell(1, horizon);
Output = cell(1, horizon);
for i=1:horizon
    Input{i} = zeros(N0,num_traj);
    Output{i} = zeros(Nend,num_traj);
end

for j = 1:num_traj
    
    initial=lb + rand(n,1).*(ub-lb);
    Input{1}(:,j) = initial;
    for i=1:horizon
        init_a=[Input{i}(:,j)];
        a_ego=cntr(nn, init_a);
        [~,in_out] =  ode45(@(t,x)dynamicsBench1(t,x,a_ego),[0 timestep],Input{i}(:,j)');
        in_out=in_out(end,:)';
        Output{i}(:,j)= in_out;
        if i<horizon
            Input{i+1}(:,j) = in_out;
        end
    end
    
end

theInput=Input;
theOutput=Output;
maxmin='no normalization';

end

function y = cntr(net, x)
   
    len=length(net.weights)-1;
    for i=1:len
        x=poslin(net.weights{i}*x+net.biases{i});
    end
    y=net.weights{end}*x+net.biases{end};
    
end

