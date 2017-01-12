function [Centroids, Labels, n, J, tr1, tr2, Sw, Sb] = CLP_Kmeans(DB, K, d, th)
%CLP_Kmeans Classify matrix with a K-Means algorithm
%
%   [ Centroids, Labels, n, J, tr_1, tr_2, Sw, Sb ] = CLP_Kmeans(DB, K, d)
%
%   Detailed explanation goes here

%% Initialize centroids and labels matrices

Centroids = datasample(DB, K, 2, 'Replace', false);

%% Declarar variables de qualitat
% Within-cluster scatter matrix
Sw = zeros(d,d); %afegit Héctor
ni = zeros(1,K); %afegit Héctor
Sb = zeros(d,d); %afegit Héctor % Between-cluster scatter matrix
%% Classify database
% threshold = 0.0005;

Labels = zeros(length(DB), 1);
n = 1;

J = zeros(50,1);
tr_1 = zeros(50,1); %afegit Héctor
tr_2 = zeros(50,1); %afegit Héctor



% Iterate while the cost function variates enough
condition = n <= 2;
while condition == 1

    J(n) = 0; % Cost function
    
    % Classify database
    for i = 1:length(DB)
        norms = sum(abs(repmat(...
            double(DB(:,i)), 1, K) - double(Centroids(:,:,end))).^2,1);
        [Minimum_value, Labels(i)] = min(norms);
        J(n) = Minimum_value + J(n);
    end
    
    % Update centroids
    for i = 1:K
        Centroids(:, i, n) = mean(double(DB(:, Labels==i)), 2);
    end
    
    if n > 1
        diff = J(n-1) - J(n);
        condition = (diff) > th;
    end
    
    %% Part Héctor
    for i = 1:length(DB)
        Sw = Sw + (double(DB(:,i))-double(Centroids(:,Labels(i),end)))*...
        (double(DB(:,i))-double(Centroids(:,Labels(i),end)))'; %% posant-ho a dintre del while aconsegueixo tenir un valor de la traça per a cada iteracio
        ni(Labels(i)) = ni(Labels(i)) + 1; % Add one sample to detected class
    end
    
    for j = 1:K
        m = (1/length(DB))*ni*double(Centroids(:,:,end))';
        Sb = Sb + ni(j)*(double(Centroids(:,j,end))-m')*(double(Centroids(:,j,end))-m')';
    end

    % Total scatter matrix
    St = Sb+Sw;

    % Trace metrics
    tr_1 (n) = trace(St\Sw);
    tr_2 (n) = trace(Sw\Sb);

    %%
    n = n+1;
end

% Take actual values of J,tr_1 and tr_2 (eliminate the rests of preallocated data)
J = J(1:n-1);
tr1 = tr_1(1:n-1);
tr2 = tr_2(1:n-1); 


end