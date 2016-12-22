%% Main script of the K-Means classifier
clear

% Parameters
L = 4;
N = 10000;
d = 2;

probabilitat = rand(L,1);
probabilitat = probabilitat./sum(probabilitat);

[DB, Nnew] = CLP_Generate(L,N,d,probabilitat);

% Draw clusters
scatter(DB(1,:), DB(2,:)), hold on

%% Classify with K-Means clustering
K = 4;
[Centroides, Labels, n] = CLP_Kmeans(DB(1:d, :),K, d);

%% Plot results 
% Plot the evolution of the centroids
for i=1:K
    c = rand(1,3);
    scatter(Centroides(1,i,:), Centroides(2,i,:), 'x', 'MarkerEdgeColor', c/sum(c))
    line(reshape(Centroides(1,i,:),[1,n]), reshape(Centroides(2,i,:),[1,n]), 'Color', c/sum(c))
end
hold off

% Plot DB with labeling
figure, hold on
for i=1:K
    c = rand(1,3);
    scatter(DB(1,Labels==i), DB(2,Labels==i), 'MarkerEdgeColor', c/sum(c))
    scatter(Centroides(1,i,:), Centroides(2,i,:), 'x', 'MarkerEdgeColor', 1 - c/sum(c))
end
hold off
