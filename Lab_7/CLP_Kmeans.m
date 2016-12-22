function [Centroides, Labels_new, n, J, traca1, traca2] = CLP_Kmeans( DB, K, d)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Initialize centroids and labels matrices
Labels = ones(length(DB), 1);

Centroides = datasample(DB, K, 2, 'Replace', false);

%% Iterate classification
Labels_new = zeros(length(DB), 1);
n=0;

while ~isequal(Labels_new, Labels)
    Labels = Labels_new;
    n=n+1;
    J(n)= 0; %funcio cost
    % Classify database
    for i = 1:length(DB)
        
        %norms = sqrt(sum(abs(repmat(DB(:,i), 1, K) - Centroides(:,:,end)).^2,1));
        norms = sum(abs(repmat(DB(:,i), 1, K) - Centroides(:,:,end)).^2,1);
        [Minim_value, Labels_new(i)] = min(norms);
        J(n)= Minim_value + J(n);
    end
    
    % Update centroids
    for i=1:K
        Centroides(:, i, n) = mean(DB(:, Labels_new==i), 2);
    end
    
    
end

Sw= zeros(d,d);
ni= zeros(1,K);
for i = 1:length(DB)
    Sw= Sw + (DB(:,i)-Centroides(:,Labels_new(i),end))*(DB(:,i)-Centroides(:,Labels_new(i),end))';
    ni(Labels_new(i))= ni(Labels_new(i)) + 1;
end

Sb = zeros(d);
for j=1:K
    m= (1/length(DB))*ni*Centroides(:,:,end)';
    Sb=Sb + ni(j)*(Centroides(:,j,end)-m)*(Centroides(:,j,end)-m)';
end
St= Sb+Sw;
traca1= trace(St\Sw);
traca2= trace(Sw\Sb);



end