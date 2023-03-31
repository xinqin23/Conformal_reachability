function [theInput, theOutput, maxmin] = ACC_nln_Datagenerator_ss(lb, ub, nn, timestep, normalization, num_traj , horizon) 

n=6;


E=[ 0  0  0  0  1  0;...
    1  0  0 -1  0  0;...
    0  1  0  0 -1  0];

V_set=30;
t_gap=1.4;


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
        init_a=[V_set; t_gap; E*Input{i}(:,j)];
        a_ego=cntr(nn, init_a);
        [~,in_out] =  ode45(@(t,x)dynamicsACC(t,x,a_ego),[0 timestep],Input{i}(:,j)');
        in_out=in_out(end,:)';
        Output{i}(:,j)= in_out;
        if i<horizon
            Input{i+1}(:,j) = in_out;
        end
    end
    
end

if normalization==1
    
    a=-1;
    b=1;
    maxmin = cell(1, horizon);
    for i=1:horizon
        maxin = max(Input{i}')';
        maxmin{i}.maxin=maxin;
        minin = min(Input{i}')';
        maxmin{i}.minin=minin;
        maxout= maxin;
        minout= minin;
        theInput{i} = (b-a) * diag(1./ (maxin-minin) ) * ( Input{i} - minin )  + a ;
        theOutput{i}= (b-a) * diag(1./(maxout-minout)) * (Output{i} - minout)  + a ;
    end
elseif normalization==0
    theInput=Input;
    theOutput=Output;
    maxmin='no normalization';
end
end




function y = cntr(net, x)
   
    len=length(net.weights)-1;
    for i=1:len
        x=poslin(net.weights{i}*x+net.biases{i});
    end
    y=net.weights{end}*x+net.biases{end};
    
end
    