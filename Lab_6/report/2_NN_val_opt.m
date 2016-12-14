%% Validacion de numero de neuronas en capa oculta
if i_valida_hidden==1;
    
    MaxLayerSize = 15;
    opt_val_error = zeros(MaxLayerSize, 1);
    
    % Generate Train, Val & Test BDs and Labels
    % Prepair Train targets
    for hiddenLayerSize = 1:MaxLayerSize        
        % Create a Pattern Recognition Network
        net = patternnet(hiddenLayerSize);
        net.performFcn='mse';
        %net.trainFcn='trainscg';  % Conjugate gradient
        %     net.trainFcn='traingd';   %Back Propagation
        net.trainFcn='trainlm';   %Levenberg-Marquadt
        net = configure(net,Brain_Etiq',Target_BD);
        
        net.divideFcn='divideind'; % The database is divided by indices
        net.divideParam.trainInd=Index_train;
        net.divideParam.valInd=Index_val;
        net.divideParam.testInd=Index_test;
        %         net.divideParam.trainRatio = 1;
        %         net.divideParam.valRatio = 0;
        %         net.divideParam.testRatio = 0;
        
        net.trainParam.epochs = 100;
        %net.trainParam.max_fail=round(net.trainParam.epochs/10); % Can set the
        %number of consecutive high values of the error over epochs in the validation set.
        %Used to stop the training.
        
        net = train(net,Brain_Etiq',Target_BD);% Train the Network
        
        % Measure val error
        outputs = net(X_val');
        [~, Index_out]=max(outputs);
        opt_val_error(hiddenLayerSize) = length(find(Labels_val~=Index_out'))/length(Labels_val);
        %         fprintf(1,' error NN val = %g   \n', opt_val_error(hiddenLayerSize));
        %         CM_Val=confusionmat(Labels_val,Index_out);
    end
    % Find optimal layer size
    [~, OptimalLayerSize] = min(opt_val_error);
    
    % Create a Pattern Recognition Network from OPTIMAL SIZE
    net = patternnet(hiddenLayerSize);
    net.performFcn='mse';
    %net.trainFcn='trainscg';  % Conjugate gradient
    %     net.trainFcn='traingd';   %Back Propagation
    net.trainFcn='trainlm';   %Levenberg-Marquadt
    net = configure(net,Brain_Etiq',Target_BD);
    
    net.divideFcn='divideind'; % The database is divided by indices
    net.divideParam.trainInd=Index_train;
    net.divideParam.valInd=Index_val;
    net.divideParam.testInd=Index_test;
    
    net.trainParam.epochs = 1000;
    %net.trainParam.max_fail=round(net.trainParam.epochs/10); % Can set the
    %number of consecutive high values of the error over epochs in the validation set.
    %Used to stop the training.
    
    net = train(net,Brain_Etiq',Target_BD);% Train the Network
    
    %% Plot train, val and test errors with the number of hidden neurons
    % Measure Train error
    outputs = net(X_train');
    [~, Index_out]=max(outputs);
    NN_Error_train=length(find(Labels_train~=Index_out'))/length(Labels_train);
    fprintf(1,' OPTIMAL error NN train = %g   \n', NN_Error_train);
    CM_Train=confusionmat(Labels_train,Index_out)
    % Measure val error
    outputs = net(X_val');
    [~, Index_out]=max(outputs);
    NN_Error_val=length(find(Labels_val~=Index_out'))/length(Labels_val);
    fprintf(1,' OPTIMAL error NN val = %g   \n', NN_Error_val);
    CM_Val=confusionmat(Labels_val,Index_out)
    % Measure Test error
    outputs = net(X_test');
    [~, Index_out]=max(outputs);
    NN_Error_test=length(find(Labels_test~=Index_out'))/length(Labels_test);
    fprintf(1,' OPTIMAL error NN test = %g   \n', NN_Error_test);
    CM_Test=confusionmat(Labels_test,Index_out)
    
end
