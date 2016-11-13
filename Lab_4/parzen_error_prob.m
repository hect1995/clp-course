% Error probabilities in Parzen predictions
%% Load data
load('Predict_test.mat')
load('Predict_train.mat')
load('Labels_test.mat')
load('Labels_train.mat')

%% Replicate Labels
Labels_test = repmat(Labels_test, 1, 4);
Labels_train = repmat(Labels_train, 1, 4);

%% Compute error probability
Train_prob= 1 - (sum(Labels_train==Predict_train)/length(Labels_train));
Test_prob= 1 - (sum(Labels_test==Predict_test)/length(Labels_test));
