% Codigo usado en el ejercicio 2.5
% Hipotesis clase 1
index=find(Labels==1);

[h1,p1,stats1] = chi2gof(X(index,1),'Alpha',0.001);
df1 = stats1.df;

% Hipotesis clase 2
index=find(Labels==2);

[h2,p2,stats2] = chi2gof(X(index,1),'Alpha',0.001);
df2 = stats2.df;

% Hipotesis clase 3
index=find(Labels==3);

[h3,p3,stats3] = chi2gof(X(index,1),'Alpha',0.001);
df3 = stats3.df;