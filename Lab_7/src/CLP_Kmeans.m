function [Centroids, Labels, n, J, tr1, tr2, Sw, Sb] = CLP_Kmeans(DB, K, d, th)
%CLP_Kmeans Classify matrix with a K-Means algorithm
%
%   [ Centroids, Labels, n, J, tr_1, tr_2, Sw, Sb ] = CLP_Kmeans(DB, K, d)
%
%   Detailed explanation goes here

%% Initialize centroids and labels matrices

Centroids = datasample(DB, K, 2, 'Replace', false);

%% Classify database
% threshold = 0.0005;

Labels = zeros(length(DB), 1);
n = 1;

J = zeros(50,1);


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
    
    n = n+1;
end

% Take actual values of J (eliminate the rests of preallocated data)
J = J(1:n-1);

%% Compute error metrics
% Within-cluster scatter matrix
Sw = zeros(d,d);
ni = zeros(1,K);

for i = 1:length(DB)
    Sw = Sw + (double(DB(:,i))-double(Centroids(:,Labels(i),end)))*...
        (double(DB(:,i))-double(Centroids(:,Labels(i),end)))';
    ni(Labels(i)) = ni(Labels(i)) + 1; % Add one sample to detected class
end

% Between-cluster scatter matrix
Sb = zeros(d,d);

for j = 1:K
    m = (1/length(DB))*ni*double(Centroids(:,:,end))';
    Sb = Sb + ni(j)*(double(Centroids(:,j,end))-m')*(double(Centroids(:,j,end))-m')';
end

% Total scatter matrix
St = Sb+Sw;

% Trace metrics
tr1 = trace(St\Sw);
tr2 = trace(Sw\Sb);

end