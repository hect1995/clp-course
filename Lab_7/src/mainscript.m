%% Exercise 2.1 script of the K-Means classifier
clear
close all

% Switch to activate or deactivate the plotting of all classifiers
plot_clusters = 1;

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

%% Section 3
% Classify with K-Means clustering

% Preallocate results from the classifier
J = cell(9,1);
minimized_J = zeros(9,1);

trace1 = zeros(9,1);
trace2 = zeros(9,1);

Sw = zeros(2,2,9);
Sb = zeros(2,2,9);

for K=2:10
    [Centroides, Labels, n , J{K-1}, trace1(K-1), trace2(K-1), ...
        Sw(:,:,K-1), Sb(:,:,K-1)] = CLP_Kmeans(DB(1:d, :),K, d);
    
    minimized_J(K-1) = J{K-1}(end);
    
    if plot_clusters
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
end

%% Section 4
% Analyze classification metrics

figure, hold on
plot(2:10, minimized_J)
grid on
title('J function')
hold off

figure, hold on
plot(2:10, trace1)
grid on
title('Trace 1')
hold off

figure, hold on
plot(2:10, trace2)
grid on
title('Trace 2')
hold off

% Trace 2 is great, Trace 1 not good, because it decays constantly. It does so
% because as we increase the number of clusters, each of them is more compact,
% so the "within" metric improves.
%
% The J function behaves sometimes roughly like Trace 1
%
% We want to minimize Trace 1 and maximize Trace 2
%
% Trace 2 increases greatly when different clusters are classified as such,
% while Trace 1 does slightly decrease in the same situation.
%
% Because of this, Trace 2 is the best metric
