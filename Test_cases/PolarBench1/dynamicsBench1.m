function [dx]=dynamicsBench1(t,x,u)


% x1 = lead_car position
% x2 = lead_car velocity
% x3 = lead_car internal state

% x4 = ego_car position
% x5 = ego_car velocity
% x6 = ego_car internal state

dx(1,1)=x(2);

dx(2,1) = u*x(2)^2 - x(1);

