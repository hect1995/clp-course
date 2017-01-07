%% Exercise 2.3 script of the K-Means classifier
% Classify a new database

close all
clear

heatmap = 1;
plot_clusters = 1;

%% Parse Iris database

file_path = 'db/bezdekIris.data';
[ DB, db_labels ] = CLP_Parse_DB( file_path );

d = size(DB, 1);

%% Classify DB with K-means

K = 3;
th = 0.0005;

[ Centroids, Labels, n, J, tr_1, tr_2, Sw, Sb ] = CLP_Kmeans(DB, K, d, th);

%% Plot results
% There will be 2 figures, one for petals and one for sepals. This separation is
% necessary, as the DB is 4-dimensional

close all

if plot_clusters
    
    % Plot DB with color labeling
    c = rand(K,3); % Use the same colours for the two figures
    
    figure
    subplot(2,2,1)
    hold on
    for i=1:K
        scatter(DB(1,Labels==i), DB(2,Labels==i), ...
            'MarkerEdgeColor', c(i,:)/sum(c(i,:)),...
            'MarkerFaceColor', c(i,:)/sum(c(i,:)))
        scatter(Centroids(1,i,:), Centroids(2,i,:), ...
            'x', 'MarkerEdgeColor', 1 - c(i,:)/sum(c(i,:)))
    end
    
    title('Sepal length vs Sepal width')
    xlabel('Sepal length (cm)')
    ylabel('Sepal width (cm)')
    hold off
    
    subplot(2,2,2)
    hold on
    for i=1:K
        scatter(DB(3,Labels==i), DB(4,Labels==i), ...
            'MarkerEdgeColor', c(i,:)/sum(c(i,:)),...
            'MarkerFaceColor', c(i,:)/sum(c(i,:)))
        scatter(Centroids(3,i,:), Centroids(4,i,:), ...
            'x', 'MarkerEdgeColor', 1 - c(i,:)/sum(c(i,:)))
    end
    
    title('Petal length vs Petal width')
    xlabel('Petal length (cm)')
    ylabel('Petal width (cm)')
    hold off
end

% Display accuracy of the classification as heatmaps
if heatmap
    % Colormap of database labels
    subplot(2,2,3)
    colormap('hot')
    imagesc(db_labels * db_labels')
    colorbar
    title('Database labels heatmap')
    
    % Colormap of classifier labels
    subplot(2,2,4)
    colormap('hot')
    imagesc(Labels * Labels')
    colorbar
    title('K-Means labels heatmap')
    
    % If the classifier was perfect, the 9 squares in the K-Means would look
    % like the DB squares (although the order of the colors can be different)
    %
    % If there are any lines in the classifier heatmap, those lines indicate
    % misclassified vectors
end
