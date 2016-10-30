% Compute KNN for k=1:10

errors = zeros(10,2);

for k=1:10
%     aux_errors = zeros(10,2);
    
%     for i=1:10
       [errors(k,:)] = compute_knn(1,4,k);
%     end
    
%     errors(k,:) = mean(aux_errors, 1);
end

plot(1:10, errors(:,1), 1:10, errors(:,2)), hold on
grid on
legend('Train Error', 'Test Error', 'best');

hold off
