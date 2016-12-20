function  [DB,Nnew]  = CLP_Generate( L,N,d,probabilitat )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%% Crear mu
values = round(N*probabilitat); %obtenim numero total de mostres per a cada cluster
Nnew = sum(values);

matriu_mitjes = zeros(d,L);
matriu_sigma = zeros(d,d,L);

DB = zeros(d +1,Nnew); %la ultima fila és per els labels 

a = -10;
b = 10;

index = 1;

for i=1:L
    matriu_mitjes(:,i) = (b-a).*rand(d,1) + a;
    matriu_sigma(:,:,i) = diag(rand(d,1));
    DB(1:d, index:index+values(i)-1) = mvnrnd(matriu_mitjes(:,i), matriu_sigma(:,:,i),values(i))';
    DB(d+1, index:index+values(i)-1) = i;
    index = index+values(i);
end

% Shuffle database columns
 DB=DB(:,randperm(length(DB)));
end

