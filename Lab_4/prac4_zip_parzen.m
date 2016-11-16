function [Iteration_prob] = prac4_zip_parzen()
% updated to Matlab2015
% MC 2016
% close all;
% clear;
% clc;
%OPCIONES
i_dib=0;					%0 NO /1 SI: DIBUJOS DE DIGITOS
i_CM=0;						%0 NO /1 SI: CALCULA MATRIZ DE CONFUSION
N_classes=10;
K_neig=10;                      %PARAMETRO K en knn
%% Elecci�n de la transformada y reducci�n de dimensi�n
% disp(' ')
% disp('Elegir Transformada')
% i_transform=input(' No transformar (0), DCT (1)  Hadamard (2) = ');
% if i_transform >0
%     disp(' ')
%     disp('Elegir Dimensi�n Reducida')
%     N_dim=input(' Dim =  ');
% else
%     N_dim=16;
% end
i_transform=1;
N_dim=4;
N_feat=N_dim*N_dim;
%% Lectura BD de train
X_train=[];             % Matriz de Nx256 que contiene todos los vectores
% Cada muestra va entre 0(Negro) y 1(Blanco)
Labels_train=[];        % Etiquetas (Inicialmente los datos estan ordenados
% por clases del 0 al 9)
for k=0:N_classes-1
    nombre=sprintf('train%d.txt',k);
    [data] = textread(nombre,'','delimiter',',');
    %data=round(data);  %OPCIONAL elimina los grises
    %y lo deja todo en blanco y negro
    X_train=[X_train;data];
    N_size=size(data);
    Labels_train=[Labels_train;k*ones(N_size(1),1)];
end
clear nombre data N_size k

%% Lectura BD de test
nombre=sprintf('zip.test');
[data] = textread(nombre,'','delimiter',' ');
Labels_test =data(:,1);
X_test=data(:,2:size(data,2));
clear nombre data


%% stratified data partitioning
X=[X_train; X_test];
Labels=[Labels_train; Labels_test];
P_train=0.5;  % Train and test sizes are equal
Index_train=[];
Index_test=[];
for i_class=0:N_classes-1
    index=find(Labels==i_class);
    N_i_class=length(index);
    [I_train,I_test] = dividerand(N_i_class,P_train,1-P_train);
    Index_train=[Index_train;index(I_train)];
    Index_test=[Index_test;index(I_test)];
end
% Train Selection and mixing
X_train=X(Index_train,:);
Labels_train=Labels(Index_train);
% Test Selection and mixing
X_test=X(Index_test,:);
Labels_test=Labels(Index_test);
clear Index_train Index_test index i_class N_i_class I_train I_test

%% OPCION TRANSFORMADAS
A2=hadamard(N_dim);
if i_transform >0
    % Transformamos BD de train
    A=hadamard(16);
    N_d2=floor(16/N_dim);
    N_samples=size(X_train,1);
    X_aux=zeros(size(X_train,1),N_feat);
    if i_transform==1
        % DCT
        for i_samples=1:N_samples
            data=X_train(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=dct2(data);
            data=data(1:N_dim,1:N_dim);
            X_aux(i_samples,:)=data(:)';
        end
    else
        % Hadamard
        for i_samples=1:N_samples
            data=X_train(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=A*data*A';
            data=data(1:N_d2:16,1:N_d2:16);
            X_aux(i_samples,:)=data(:)';
        end
    end
    
    if i_dib==1
        figure('name','Dominio Transformado')
        for k=0:N_classes-1
            subplot(3,4,k+1)
            ind=find(Labels_train==k);
            N_ale=randi(length(ind));
            data=X_aux(ind(N_ale),:);
            data=reshape(data,N_dim,N_dim);
            imagesc(abs(data));
            colorbar
            xlabel(k)
        end
        clear N_ale ind k data
    end
    X_train=X_aux;
    
    %Transformamos BD de test
    N_samples=size(X_test,1);
    X_aux=zeros(size(X_test,1),N_feat);
    if i_transform==1
        % DCT
        for i_samples=1:N_samples
            data=X_test(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=dct2(data);
            data=data(1:N_dim,1:N_dim);
            X_aux(i_samples,:)=data(:)';
        end
    else
        % Hadamard
        for i_samples=1:N_samples
            data=X_test(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=A*data*A';
            data=data(1:N_d2:16,1:N_d2:16);
            X_aux(i_samples,:)=data(:)';
        end
    end
    X_test=X_aux;
    clear X_aux N_samples A i_samples N_d2
end

%% OPCION dibujos de imagenes
if i_dib==1
    figure('name','Images TRAIN')
    for k=0:N_classes-1
        subplot(2,5,k+1)
        ind=find(Labels_train==k);
        N_ale=randi(length(ind));
        data=X_train(ind(N_ale),:);
        data=reshape(data,N_dim,N_dim);
        if i_transform==1
            data=idct2(data);
        elseif i_transform==2
            data=A2*data*A2';
        else
            data=data';
        end
        imagesc(1-data);
        %data=round(data);  %OPCIONAL elimina los grises
        % y lo deja todo en blanco y negro
        ylabel(k)
    end
    colormap(gray)
    
    figure('name','Images Test')
    for k=0:N_classes-1
        subplot(2,5,k+1)
        ind=find(Labels_test==k);
        N_ale=randi(length(ind));
        data=X_test(ind(N_ale),:);
        data=reshape(data,N_dim,N_dim);
        if i_transform==1
            data=idct2(data);
        elseif i_transform==2
            data=A2*data*A2';
        else
            data=data';
        end
        imagesc(1-data);
        ylabel(k)
    end
    colormap(gray)
    clear N_ale k data ind N_ale
end
clear i_dib A2 i_transform

%% Create a Parzen classifier
% h = [1, 10, 20, 100];
h = 10:20;

Predict_train = zeros(length(X_train), length(h));
Predict_test = zeros(length(X_test), length(h));
for i = 1:length(h)
    Predict_train(:, i) = predict_parzen(X_train,Labels_train,N_classes,h(i),X_train);
    Predict_test(:, i) = predict_parzen(X_train,Labels_train,N_classes,h(i),X_test);
end

%% Replicate Labels
Labels_test = repmat(Labels_test, 1, length(h));
Labels_train = repmat(Labels_train, 1, length(h));

%% Compute error probability
% Transpose to get a column array
Train_prob = (1 - (sum(Labels_train==Predict_train)/length(Labels_train)));
Test_prob = (1 - (sum(Labels_test==Predict_test)/length(Labels_test)));

Iteration_prob = [Train_prob', Test_prob']; % Matrix to be returned
end