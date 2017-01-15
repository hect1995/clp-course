function [ DB, N_new ] = CLP_Generate( L, N, d, priori_prob )
%CLP_Generate Generate a synthetic gaussian database
%
%   [ DB, Nnew ] = CLP_Generate( L, N, d, probabilities )
%
%   This function generates L clusters of gaussian-distributed d-dimensional
%   vectors.
%
%   The L clusters contains a number of samples <= N. The total number
%   of samples (N_new) may be less than N. This could happen due to the rounding
%   process in which each cluster is assigned a number of vectors, following the
%   a priori probabilities vector (priori_prob).
%
%   INPUTS:
%   ============================================================================
%   - L: Number of clusters to be generated
%   - N: Total number of samples to be generated
%   - d: Number of dimensions in the vectors
%   - priori_prob: Vector of a priori probabilities for the clusters
%
%   Notice that L == length(priori_prob)
%
%   OUTPUTS:
%   ============================================================================
%   - DB: Matrix containing the generated database (first d rows contain the
%         vector components, last row contains the labels)
%   - N_new: Actual total number of generated vectors (N_new <= N)
%
%   TODO: Insert license notice

%% Check the input parameters for errors
assert(length(priori_prob) == L,strcat('Input parameter size mismatch.',...
    'Please, make sure the vector priori_prob has L parameters'))

%% Initialize parameters
% Compute the number of samples in each cluster
values = round(N*priori_prob);

% Compute the total number of samples in the generated clusters
N_new = sum(values);

% Preallocate mean and variance matrices
matriu_mitjes = zeros(d,L);
matriu_sigma = zeros(d,d,L);

% The last row contains labeling information
DB = zeros(d +1,N_new);

% Initialize the range of the clusters' mean
a = -10;
b = 10;

%% Compute cluster vectors
index = 1;
for i=1:L
    % Compute means of the current cluster
    matriu_mitjes(:,i) = (b-a).*rand(d,1) + a;
    % Compute variances of the current cluster
    matriu_sigma(:,:,i) = diag(rand(d,1));
    % Get random values for the current cluster
    DB(1:d, index:index+values(i)-1) = ...
        mvnrnd(matriu_mitjes(:,i), matriu_sigma(:,:,i),values(i))';
    % Save "label" of current cluster
    DB(d+1, index:index+values(i)-1) = i;
    % Increase the index (it is used to access the columns in the DB that
    % correspond to a single cluster
    index = index+values(i);
end

% Shuffle database columns
DB=DB(:,randperm(length(DB)));
end
