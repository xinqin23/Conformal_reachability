bain = []
baout = []
for i = 1:length(Input_Data)
   bain = [bain, Input_Data{i}];
end


for i = 1:length(Output_Data)
    baout = [baout, Output_Data{i}];
end


Input = bain;
Output = baout;

save('Data.mat', 'Input', 'Output')

    