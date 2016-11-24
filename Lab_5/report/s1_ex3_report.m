%% Clasificador kernel gaussiano
    P_opt = 0;
    h_opt = 0;
    err_val_opt = Inf;
    
    P = 0.01:0.1:5;
    h = [1, 2.5, 25, 100];
    
    err_train_val = zeros(length(P), length(h), 2);
    
    for j = 1:length(P)
        for i=1:length(h)
            Gauss_model = fitcsvm(X_train, Labs_train, 'BoxConstraint',P(j),...
                'KernelFunction','RBF','KernelScale',h(i));
            fprintf(1,'\n Clasificador Kernel Gaussiano\n')
            Gauss_out = predict(Gauss_model, X_train);
            Err_train=sum(Gauss_out~=Labs_train)/length(Labs_train);
            fprintf(1,'error train = %g   \n', Err_train)
            
            Gauss_out = predict(Gauss_model, X_val);
            Err_val=sum(Gauss_out~=Labs_val)/length(Labs_val);
            fprintf(1,'error val = %g   \n', Err_val)
            fprintf(1,'\n  \n  ')
            % Test confusion matrix
            CM_Gauss_test = confusionmat(Labs_val,Gauss_out)
            
            err_train_val(j, i, 1) = Err_train;
            err_train_val(j, i, 2) = Err_val;
            
            if Err_val < err_val_opt
                err_val_opt = Err_val;
                P_opt = P(j);
                h_opt = h(i);
            end
%             clear Err_train Err_test Gauss_out
        end
    end
    
    %% Plot 3D graphic
    figure
    mesh(h, P, err_train_val(:,:,1)), hold on
    title('Training error')
    hold off
    
    figure
    mesh(h, P, err_train_val(:,:,2)), hold on
    title('Validation error')
    hold off