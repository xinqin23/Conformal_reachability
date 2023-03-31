function [theInput, theOutput, maxmin] = ACC_nln_Datagenerator_ss(lb, ub, nn, timestep, normalization, num_traj , horizon) 


inits1 = 0.8 + (0.9-0.8)*rand(100,1);
inits2 = 0.5 + 0.1*rand(100,1);
% [0.8, 0.9] to [0.5, 0.6]
%  a + (b-a).*rand(100,1)

inits = [inits1, inits2]
 

timestep = 0.2;

finalTime = 7;

numTrajectories = 100;

 

times = 0:timestep:finalTime;

from = cell(numTrajectories,length(times)-1);

next = cell(numTrajectories,length(times)-1);

 

for jj = 1:numTrajectories

    from{jj,1} = inits(ii,:);

%     enumerate over time steps

% at each time step pick initial condition as last timesteps "from"       

        [~,traj] = ode45(@vdp1,[0 timestep],from{});

        

% set this timesteps next to traj(end)      

end

end
    