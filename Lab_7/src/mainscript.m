%% Exercise 2.1 script of the K-Means classifier
clear

%% Section 1
% Generate DB

% Initialize parameters
L = 4;
N = 10000;
d = 2;

% Compute a priori probabilities for each cluster
probabilities = rand(L,1);
probabilities = probabilities./sum(probabilities);

% Generate DB samples
[DB, Nnew] = CLP_Generate(L,N,d,probabilities);

% Draw clusters
scatter(DB(1,:), DB(2,:))%, hold on

%% TODO: Fix Section 3 to catch all J values from CLP_Kmeans 
% Classify with K-Means clustering
J = zeros(9,1);
trace1 = zeros(9,1);
trace2 = zeros(9,1);

for K=2:10
    [Centroides, Labels, n , J(K-1), trace1(K-1), trace2(K-1), ...
        Sw(:,:,K-1), Sb(:,:,K-1)] = CLP_Kmeans(DB(1:d, :),K, d);    
    
%     % Plot evolution of the centroids
%     for i=1:K
%         c = rand(1,3);
%         scatter(Centroides(1,i,:), Centroides(2,i,:), 'x', ...
%             'MarkerEdgeColor', c/sum(c))
%         line(reshape(Centroides(1,i,:),[1,n]), ...
%             reshape(Centroides(2,i,:),[1,n]), 'Color', c/sum(c))
%     end
%     hold off

    % Plot DB with color labeling
    figure, hold on
    for i=1:K
        c = rand(1,3);
        scatter(DB(1,Labels==i), DB(2,Labels==i), ...
            'MarkerEdgeColor', c/sum(c))
        scatter(Centroides(1,i,:), Centroides(2,i,:), ...
            'x', 'MarkerEdgeColor', 1 - c/sum(c))
    end
end

figure
semilogy(2:10, J), grid on

%% TODO Section 4
% Analyze classification metrics
