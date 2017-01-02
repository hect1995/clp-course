function [ Centroids, Labels, n, J, tr_1, tr_2, Sw, Sb ] = CLP_Kmeans(DB, K, d)
%CLP_Kmeans Classify matrix with a K-Means algorithm
%
%   [ Centroids, Labels, n, J, tr_1, tr_2, Sw, Sb ] = CLP_Kmeans(DB, K, d)
%
%   Detailed explanation goes here

%% Initialize centroids and labels matrices
Labels_old = ones(length(DB), 1);

Centroids = datasample(DB, K, 2, 'Replace', false);
% % Initialize centroids randomly
% a = -5;
% b = 5;
% original_Centroids = a + (b-a).*rand(d, K);
% Centroids = original_Centroids;

%% Classify database
threshold = 0.0005;

Labels = zeros(length(DB), 1);
n = 1;

J_aux = zeros(50,1);

% TODO change condition to a measure of the variation of J
% while ~isequal(Labels, Labels_old) % Iterate until no change in lavels

condition = n <= 2;

% while (J_aux(n-1) - J_aux(n)) > threshold || n == 0
while condition == 1
    
    Labels_old = Labels;

    J_aux(n) = 0; % Cost function
    
    % Classify database
    for i = 1:length(DB)
        norms = sum(abs(repmat(...
            double(DB(:,i)), 1, K) - double(Centroids(:,:,end))).^2,1);
        [Minimum_value, Labels(i)] = min(norms);
        J_aux(n) = Minimum_value + J_aux(n);
    end
    
    % Update centroids
    for i = 1:K
        Centroids(:, i, n) = mean(double(DB(:, Labels==i)), 2);
    end
    
    if n > 1
        diff = J_aux(n-1) - J_aux(n);
        condition = (diff) > threshold;
    end
    
    n = n+1;
end

% Take actual values of J_aux (eliminate the rests of preallocated data)
J_aux = J_aux(1:n-1);

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
tr_1 = trace(St\Sw);
tr_2 = trace(Sw\Sb);

% Take the J value of the last iteration
J = J_aux(end);

end