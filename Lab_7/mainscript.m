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
scatter(DB(1,:), DB(2,:))%, hold on

%% Classify with K-Means clustering
J = zeros(10,1);
trace1 = zeros(10,1);
trace2 = zeros(10,1);

for K=2:10
    [Centroides, Labels, n , J(K), trace1(K), trace2(K), Sw(:,:,K-1), Sb(:,:,K-1)] = CLP_Kmeans(DB(1:d, :),K, d);
    % [Centroides, Labels, n , J] = CLP_Kmeans(DB(1:end, :),K, d);
    
    
    %% Plot results
    % Plot the evolution of the centroids
    % for i=1:K
    %     c = rand(1,3);
    %     scatter(Centroides(1,i,:), Centroides(2,i,:), 'x', 'MarkerEdgeColor', c/sum(c))
    %     line(reshape(Centroides(1,i,:),[1,n]), reshape(Centroides(2,i,:),[1,n]), 'Color', c/sum(c))
    % end
    % hold off
    
%     % Plot DB with labeling
%     figure, hold on
%     for i=1:K
%         c = rand(1,3);
%         scatter(DB(1,Labels==i), DB(2,Labels==i), 'MarkerEdgeColor', c/sum(c))
%         scatter(Centroides(1,i,:), Centroides(2,i,:), 'x', 'MarkerEdgeColor', 1 - c/sum(c))
%     end
%     hold off
end
%% CUANTIFICAR IMATGE
image= imread('lena.jpg');
color1= reshape(image(:,:,1),1,[]);
color2= reshape(image(:,:,2),1,[]);
color3= reshape(image(:,:,3),1,[]);
imatge_rgb= [color1 ; color2 ; color3 ];
J_rgb = zeros(3,1);
trace1_rgb = zeros(3,1);
trace2_rgb = zeros(3,1);
d= 3;
K= 3;
[Centroides_rgb, Labels_rgb, n_rgb , J_rgb(K), trace1_rgb(K), trace2_rgb(K), Sw_rgb(:,:,K-1), Sb_rgb(:,:,K-1)] = CLP_Kmeans(DB(1:d, :),K, d);



%end
