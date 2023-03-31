% controller_nn = NN_Reader(0, 1, 'quad_controller_3_64')
% save('Controlquad.mat', 'controller_nn')

controller_nn = NN_Reader(4, 1, 'nn_1_relu')
save('ControlBench1.mat', 'controller_nn')
