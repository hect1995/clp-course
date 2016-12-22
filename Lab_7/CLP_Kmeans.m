function [Centroides, Labels, n] = CLP_Kmeans( DB, K, d)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %% Initialize centroids and labels matrices
    Labels = ones(length(DB), 1);
    
    Centroides = datasample(DB, K, 2, 'Replace', false);
    
    %% Iterate classification
    Labels_new = zeros(length(DB), 1);
    n=0;
    J=0; %funcio cost
    while ~isequal(Labels_new, Labels)
        Labels = Labels_new;
        n=n+1;
        
        % Classify database
        for i = 1:length(DB)
            norms = sqrt(sum(abs(repmat(DB(:,i), 1, K) - Centroides(:,:,end)).^2,1));
            [~, Labels_new(i)] = min(norms); 
            
        end
        
        % Update centroids
        for i=1:K
            Centroides(:, i, n) = mean(DB(:, Labels_new==i), 2);
        end
        
        
    end
    
    n
end