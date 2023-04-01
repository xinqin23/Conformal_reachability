function  Star_sets = ReLUplex_Reachability_ss(Center, epsilon, s2s_model, analysis_type, num_Cores, Horizon, Conf_d)

% len_model = length(s2s_model.layers);
len_model = length(s2s_model.weights)-1;
system_dim = size(Center,1);

net = Trapezius_maker_ss(s2s_model, Horizon);


%%%%%%%%%%%%%%%%%%
   
Net = net;
Star_sets = cell(1, Horizon);

   
init_star_state = Star();
init_star_state.V = [Center eye(system_dim)];
init_star_state.C = zeros(1,system_dim);
init_star_state.d = 0;
init_star_state.predicate_lb = -epsilon;
init_star_state.predicate_ub = epsilon;
init_star_state.dim = system_dim;
init_star_state.nVar = system_dim;

len = length(Net.layers);
in = init_star_state;

recur = 0;
Conf_H = cell(1, Horizon);

for i = 1:len
    
    if  mod(i, len_model)==1
        recur = recur+1;
        
        H = Star();
        H.V = [zeros(system_dim,1) eye(system_dim)];
        H.C = zeros(1,system_dim);
        H.d = 0;
        H.predicate_lb = -Conf_d(:,recur);
        H.predicate_ub = Conf_d(:,recur);
        H.dim = system_dim;
        H.nVar = system_dim;
        
        Conf_H{recur} = H;
        clear H
        
        
    end
    
    disp(['analysis of Layer ' num2str(i) ' of ' num2str(len) '.'])
    lenS = length(in);
    lenAC = 0;
    
    in_layer = in;
    
    
    if strcmp(Net.layers{i}(1), 'purelin')
        index = 1;
        dd = true;
        while dd
            index = index+1;
            if ~strcmp(Net.layers{i}(index), 'purelin')
                dd = false;
            end
        end
    else
        index = 1;
    end
    

    
    for k = 1:lenS            
        
        if mod(i, len_model)==1
            if i>1 
                the_W = blkdiag(eye(index_star-1)   ,   [eye(system_dim) ; s2s_model.weights{1}] );
                the_b = [ zeros(system_dim+index_star-1 , 1)  ;   s2s_model.biases{1}];
                in_layer(k) = affineMap(in(k), the_W, the_b);
            else
                in_layer(k) = affineMap(in(k), Net.weights{i}, Net.biases{i});
            end
        else
            in_layer(k) = affineMap(in(k), Net.weights{i}, Net.biases{i});
        end
        
        
        In_layer = in_layer(k);

        In_layer.V = In_layer.V(index:end, :);
        if size(In_layer.state_lb , 1 )~=0     
            In_layer.state_lb = In_layer.state_lb(index:end, 1);
            In_layer.state_ub = In_layer.state_ub(index:end, 1);
        end
        In_layer.dim =  size(In_layer.V , 1 );
        
        
        Layers = LayerS(eye(In_layer.dim), zeros(In_layer.dim,1) , 'poslin');
        Ln     = LayerS(eye(In_layer.dim), zeros(In_layer.dim,1) , 'purelin');
        Layers = [Layers Ln];
        
        F = FFNNS(Layers);
        %%%%    'exact-star'   or    'approx-star'
        [Out_layer, ~] = reach(F,In_layer, analysis_type, num_Cores); 
        
        lenAC_now = length(Out_layer);
        
        if mod(i,len_model)==0
            index_star = index;
            for j =1:lenAC_now
                Out_layer(j) = affineMap(Out_layer(j), s2s_model.weights{end}, s2s_model.biases{end});
                Out_layer(j) = Sum( Out_layer(j), Conf_H{recur} );
            end
            Star_sets{recur}(lenAC+1:lenAC+lenAC_now) = Out_layer; 
        end
        
        
        for j = 1:lenAC_now
            
            d_size = size(Out_layer(j).V,2)-size(in_layer(k).V(1:index-1,:), 2);
            Out_layer(j).V = [[in_layer(k).V(1:index-1,:) zeros(index-1, d_size) ] ;  Out_layer(j).V];
            
            if size(in_layer(k).state_lb , 1 )~=0
                if size(Out_layer(j).state_lb , 1 )~=0 
                    Out_layer(j).state_lb = [in_layer(k).state_lb(1:index-1,1) ; Out_layer(j).state_lb];
                    Out_layer(j).state_ub = [in_layer(k).state_ub(1:index-1,1) ; Out_layer(j).state_ub];
                end
            else
                if size(Out_layer(j).state_lb , 1 )~=0 
                    SSS =  getBox(in_layer(k));
                    Out_layer(j).state_lb = [SSS.lb(1:index-1,1) ; Out_layer(j).state_lb];
                    Out_layer(j).state_ub = [SSS.ub(1:index-1,1) ; Out_layer(j).state_ub];
                end
            end

            Out_layer(j).dim = Out_layer(j).dim + (index-1);   
            in_next(lenAC+j) = Out_layer(j);
        end
        lenAC     = lenAC+ lenAC_now;
    end
    
    clear in
    in = in_next;
    clear in_next
    
    
end



