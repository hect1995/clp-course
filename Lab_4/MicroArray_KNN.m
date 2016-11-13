% Run MA DB KNN classification
clear

%% Allocate
% CM_knn_train = zeros(8,8,4);
% CM_knn_test = zeros(8,8,4);
CM_knn_train = zeros(14,14,4);
CM_knn_test =  zeros(14,14,4);
knn_Pe_train = zeros(4,1);
knn_Pe_test = zeros(4,1);

%% Compute Pe
for k=1:4
    [knn_Pe_train(k), CM_knn_train(:,:,k), knn_Pe_test(k), CM_knn_test(:,:,k)] = prac4_MA_function(k);
end

%% Plot Pe
plot(1:4, knn_Pe_train, 1:4, knn_Pe_test), hold on
legend('P_e train', 'P_e test', 'Location', 'best')
hold off