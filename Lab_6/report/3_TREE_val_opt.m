%% Validacion de numero MAXIMO DE SPLITS - Tree
if i_valida_split==1;

    % TO DO
    d = 1;
    m = 100;
    val_error = zeros(m-d, 1);
    % Generate Train, Val & Test BDs and Labels
    for MaxNumSplits=d:m
        % Train a tree with the train BD and the train targets
        % Measure Train, Val and Test classification errors

        % Tree classifier design
        tree = fitctree(X_train,Labels_train,'MaxNumSplits',MaxNumSplits);

        % Measure Val error
        outputs = predict(tree,X_val);
        val_error(MaxNumSplits - d +1)=sum(Labels_val ~= outputs)/length(Labels_val);
        %         fprintf('\n-------------------------\n')
        %         fprintf(1,' error Tree val = %g   \n', val_error(MaxNumSplits - d +1))
        %         CM_Val=confusionmat(Labels_val,outputs);
    end

    [~, OptimalNumSplits] = min(val_error)

    % Find lowest error and its correspondent MaxNumSplits
    fprintf('\n-------------------------\n')
    fprintf(1,' Optimal Split Size = %g   \n', OptimalNumSplits)

    %% TREE IMAGE CLASSIFICATION
    outputs = predict(tree,Brain_5);
    Aux=reshape(outputs,Ndim,Ndim);
    figure('name','Optimal Tree Classified Image')
    imagesc(Aux)
    axis image
    colorbar

    % Plot train, val and test errors with the number of MaxNumSplits
    % Tree classifier design with optimal number of splits
    tree = fitctree(X_train,Labels_train,'MaxNumSplits',OptimalNumSplits);
    view(tree,'mode','graph');
    view(tree)

    % Measure Train error
    outputs = predict(tree,X_train);
    Tree_Pe_train=sum(Labels_train ~= outputs)/length(Labels_train);
    fprintf('\n------- TREE CLASSIFIER ------------------\n')
    fprintf(1,' error Tree train = %g   \n', Tree_Pe_train)
    CM_Train=confusionmat(Labels_train,outputs)
    % Measure Val error
    outputs = predict(tree,X_val);
    Tree_Pe_val=sum(Labels_val ~= outputs)/length(Labels_val);
    fprintf('\n-------------------------\n')
    fprintf(1,' error Tree val = %g   \n', Tree_Pe_val)
    CM_Val=confusionmat(Labels_val,outputs)
    % Measure Test error
    outputs = predict(tree,X_test);
    Tree_Pe_test=sum(Labels_test ~= outputs)/length(Labels_test);
    fprintf('\n-------------------------\n')
    fprintf(1,' error Tree test = %g   \n', Tree_Pe_test)
    CM_Test=confusionmat(Labels_test,outputs)
    % END TO DO
end

