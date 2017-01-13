function [Centroids, Labels, n, J, tr1, tr2, Sw, Sb] = CLP_Kmeans(DB, K, d, th)
%CLP_Kmeans Classify matrix with a K-Means algorithm
%
%   [ Centroids, Labels, n, J, tr_1, tr_2, Sw, Sb ] = CLP_Kmeans(DB, K, d)
%
%   Detailed explanation goes here

%% Initialize centroids and labels matrices

Centroids = datasample(DB, K, 2, 'Replace', false);

%% Declare quality metrics
% Within-cluster scatter matrix
Sw = zeros(d,d);
ni = zeros(1,K);

% Between-cluster scatter matrix
Sb = zeros(d,d);

%% Classify database
Labels = zeros(length(DB), 1);
n = 1;

J = zeros(50,1);
tr_1 = zeros(50,1);
tr_2 = zeros(50,1);

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

    % Compute trace metrics
    for i = 1:length(DB)
        % By computing the metrics inside the 'while' loop,
        % we can get a value for each of the iterations
        Sw = Sw + (double(DB(:,i))-double(Centroids(:,Labels(i),end)))*...
        (double(DB(:,i))-double(Centroids(:,Labels(i),end)))';

        % Add one sample to detected class
        ni(Labels(i)) = ni(Labels(i)) + 1;
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

    n = n+1;
end

%% Take actual values of J,tr_1 and tr_2
% Eliminate the unused preallocated data
J = J(1:n-1);
tr1 = tr_1(1:n-1);
tr2 = tr_2(1:n-1);


end
