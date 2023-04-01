function conf_threshold = Conf_apply(Input_Data_pre, Output_Data_pre, model, delta)

len = size(Input_Data_pre,2);
system_dim = size(model.biases{end},1);

R = zeros(system_dim, len);
residual = abs(NN(model, Input_Data_pre) - Output_Data_pre);

for j=1:system_dim
    R(j,:) = sort(residual(j,:));
end

loc = floor((len+1)*delta)+1;

if loc>len
    error('Not enough data for Conformal Inference')
end
conf_threshold = R(:,loc);

end