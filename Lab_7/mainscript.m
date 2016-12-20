clear

L= 4;
N= 10000;
d= 2;
probabilitat= rand(4,1);
probabilitat= probabilitat./sum(probabilitat);

[DB, Nnew] = CLP_Generate(L,N,d,probabilitat);

% Draw clusters
scatter(DB(1,:), DB(2,:))