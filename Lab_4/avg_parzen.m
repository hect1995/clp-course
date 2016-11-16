%% Compute Parzen error probabilities 5 times and average
errors = zeros(11,2,5);

parfor i=1:5
    i
    
    [errors(:,:,i)] = prac4_zip_parzen;
end

mean_errors = mean(errors, 3);

%% Plot results
plot(10:20, mean_errors(:,1), 10:20, mean_errors(:,2)), hold on
grid on
legend('Train Error', 'Test Error', 'Location', 'best');

hold off

save('parzen_10_20_5reps_data.mat', 'errors')