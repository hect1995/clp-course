%% Compute KNN for k=1:10

errors = zeros(10,2,10);

for i=1:10
       [errors(:,:,i)] = compute_knn(1,4);
end

mean_errors = mean(errors, 3);

%% Plot results
plot(1:10, mean_errors(:,1), 1:10, mean_errors(:,2)), hold on
grid on
legend('Train Error', 'Test Error', 'Location', 'best');

hold off
