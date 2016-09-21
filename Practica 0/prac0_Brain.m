%  PRAC0
%  MC Feb. 2016
% Lectura de la base de datos BRAIN
% Análisis de Gaussianidad: Gráfico y Análitico

%% OPCIONES
clear
close all
i_histfit=1;                    %0 NO /1 SI: HISTOGRAMAS
i_cdfplot=1;				    %0 NO /1 SI: calcula cdf
i_scplot=1;					    %0 NO /1 SI: SCATTERPLOT DE CARACTERISTICAS
i_kurskew=1;				    %0 NO /1 SI: calcula Kurtosis Y Skewness
i_plotnorm=1;				    %0 NO /1 SI: calcula PLOTNORM

%%  DIBUJAR IMAGENES DE LA BASE BRAIN
load PRBB_Brain2
N_coor=5;
N_dim=256;
N_dim2=N_dim*N_dim;
Brain=Brain(:,1:N_coor);    % No se utilizan las 3 últimas columnas

figure
for i1=1:N_coor
    subplot (3, 2, i1);
    Vaux=Brain(:,i1);
    Aux=reshape(Vaux,N_dim,N_dim);
    imagesc(Aux);
    axis image
    colorbar
end
clear Vaux

%%  CARGAR UNA VERSION RECORTADA DE LAS CINCO IMAGENES
load Brain_Prac0
X=Signal;
clear Signal
[N_datos,N_feat]=size(X);
C_Clases=3;       % 1 (Grey) 2(White) 3 (Liquido cefaloraquideo);
                  % No estamos utilizando la clase "fondo"


%%  KURTOSIS-SKEWNESS
if i_kurskew==1
    figure('name','Kurtosis,Skewness')
    index=find(Labels==1);
    subplot(2,3,1)
    bar(kurtosis(X(index,:))-3)
    ylabel('KURTOSIS')
    title('Grey')
    grid
    subplot(2,3,4)
    bar(skewness(X(index,:)))
    ylabel('SKEWNESS')
    grid
 
    index=find(Labels==2);
    subplot(2,3,2)
    bar(kurtosis(X(index,:))-3)
    title('White')
    grid
    subplot(2,3,5)
    bar(skewness(X(index,:)))
    grid
    
    index=find(Labels==3);
    subplot(2,3,3)
    bar(kurtosis(X(index,:))-3)
    title('Liq Cef')
    grid
    subplot(2,3,6)
    bar(skewness(X(index,:)))
    grid 
end

%% SCATTER PLOT

if i_scplot==1
    varNames = {'feat 1' 'feat 2' 'feat 3' 'feat 4' 'feat 5'};
    figure('name','Scatter Plot')
    gplotmatrix(X,X,Labels,'bgr',[],[],'on','hist',varNames,varNames)
    zoom on
end

%% HISTOGRAMA
i_class=1 % Clase que se analiza 1 (Grey) 2(White) 3 (Liquido cefaloraquideo);
if i_histfit==1
    figure('name','Histograma')
    index=find(Labels==i_class);  %Seleccionar clase a analizar
    for i_feat=1:N_feat
        subplot(3,2,i_feat)
        histfit(X(index,i_feat))
        grid
        zoom on
        title(i_feat)
    end
end

%% cdf
if i_cdfplot==1
    figure('name','cdfplot')
    index=find(Labels==i_class);
    for i_feat=1:N_feat
        subplot(3,2,i_feat)
        cdfplot(X(index,i_feat))
        title(i_feat)
        zoom on
    end
end

%%  PLOTNORM
if i_plotnorm==1
    figure('name','Plotnorm')
    index=find(Labels==i_class);
    for i_feat=1:N_feat
        subplot(3,2,i_feat)
        qqplot(X(index,i_feat))
        grid
        title(i_feat)
        zoom on
    end
end


