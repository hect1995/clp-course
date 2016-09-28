% Codigo usado en el ejercicio 2.4  
% Intervalos clase 1
index=find(Labels==1);

alpha95 = 1 - 0.95;
alpha99 = 1 - 0.99;
alpha999 = 1 - 0.999;

x = mean(X(index,1));
s2 = var(X(index,1));
n = length(index);
t95 = tinv(1-(alpha95/2), n-1);
t99 = tinv(1-(alpha99/2), n-1);
t999 = tinv(1-(alpha999/2), n-1);

interval_p_1_95 = x + t95 * (sqrt(s2/n));
interval_n_1_95 = x - t95 * (sqrt(s2/n));
interval_p_1_99 = x + t99 * (sqrt(s2/n));
interval_n_1_99 = x - t99 * (sqrt(s2/n));
interval_p_1_999 = x + t999 * (sqrt(s2/n));
interval_n_1_999 = x - t999 * (sqrt(s2/n));
    
% Intervalos clase 2
index=find(Labels==2);

x = mean(X(index,1));
s2 = var(X(index,1));
n = length(index);
t95 = tinv(1-(alpha95/2), n-1);
t99 = tinv(1-(alpha99/2), n-1);
t999 = tinv(1-(alpha999/2), n-1);

interval_p_2_95 = x + t95 * (sqrt(s2/n));
interval_n_2_95 = x - t95 * (sqrt(s2/n));
interval_p_2_99 = x + t99 * (sqrt(s2/n));
interval_n_2_99 = x - t99 * (sqrt(s2/n));
interval_p_2_999 = x + t999 * (sqrt(s2/n));
interval_n_2_999 = x - t999 * (sqrt(s2/n));

% Intervalos clase 3
index=find(Labels==3);

x = mean(X(index,1));
s2 = var(X(index,1));
n = length(index);
t95 = tinv(1-(alpha95/2), n-1);
t99 = tinv(1-(alpha99/2), n-1);
t999 = tinv(1-(alpha999/2), n-1);

interval_p_3_95 = x + t95 * (sqrt(s2/n));
interval_n_3_95 = x - t95 * (sqrt(s2/n));
interval_p_3_99 = x + t99 * (sqrt(s2/n));
interval_n_3_99 = x - t99 * (sqrt(s2/n));
interval_p_3_999 = x + t999 * (sqrt(s2/n));
interval_n_3_999 = x - t999 * (sqrt(s2/n));