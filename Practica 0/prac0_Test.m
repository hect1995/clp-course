%Distribuciones Sinteticas
%OPCIONES
clear
close all
i_histfit=1;                     %0 NO /1 SI: HISTOGRAMAS
i_kurskew=1;				        %0 NO /1 SI: calcula Kurtosis Y Skewness
i_cdfplot=1;				        %0 NO /1 SI: calcula cdf
i_qqplot=1;				        %0 NO /1 SI: calcula QQ-plot
%% GENERACION DE DISTRIBUCIONES
N=10000;
N_points=50; %Puntos de visualizacion de histogramas.
rand('state',sum(100*clock))

% DISTRIBUCION GAUSSIANA
Y=5*randn(1,N)+3*ones(1,N);

% DISTRIBUCION RAYLEIGH
Y_abs=abs(5*randn(1,N)+5*1i*randn(1,N));

% DISTRIBUCION UNIFORME
rand('state',sum(100*clock))
Z=rand(1,N);

% DISTRIBUCION LAPLACIANA
i_1=find(Z<0.5);
i_2=find(Z>0.5);
Z_Lap(i_1)=sqrt(0.5)*log(2*Z(i_1));
Z_Lap(i_2)=-sqrt(0.5)*log(2*(1-Z(i_2)));

Z=Z-0.5; %Se elimina la media de la uniforme.

% MATRIZ DE DISTRIBUCIONES
XX=[Y;Y_abs;Z_Lap;Z];

%% HISTOGRAMAS
if i_histfit==1
    figure('name','Histogramas')
    subplot(2,2,1)
    histfit(Y,N_points)
    title('Normal Distribution')
    grid
    subplot(2,2,2)
    histfit(Y_abs,N_points)
    title('Rayleigh Distribution')
    grid
    subplot(2,2,3)
    histfit(Z_Lap,N_points)
    title('Laplacian Distribution')
    grid
    subplot(2,2,4)
    histfit(Z,N_points)
    title('Uniform Distribution')
    grid
end


%% cdfplot
if i_cdfplot==1
    figure('name','cdfplot')
    subplot(2,2,1)
    cdfplot(Y)
    title('Normal Distribution')
    subplot(2,2,2)
    cdfplot(Y_abs)
    title('Rayleigh Distribution')
    subplot(2,2,3)
    cdfplot(Z_Lap)
    title('Laplacian Distribution')
    subplot(2,2,4)
    cdfplot(Z)
    title('Uniform Distribution')
end

%% KURTOSIS-SKEWNESS
if i_kurskew==1
    figure('name','KURTOSIS-SKEWNESS')
    subplot(1,2,1)
    bar(kurtosis(XX')-3)
    grid
    title('KURTOSIS')
    subplot(1,2,2)
    bar(skewness(XX'))
    grid
    title('SKEWNESS')
end

%% PLOTNORM
if i_qqplot==1
    % Es como el norm plot
    % pero perimite comparar dos distribuciones diferentes entre si
    figure('name','qqplot=normplot')
    subplot(2,2,1)
    qqplot(Y)
    grid
    title('Normal Distribution')
    subplot(2,2,2)
    qqplot(Y_abs)
    grid
    title('Rayleigh Distribution')
    subplot(2,2,3)
    qqplot(Z_Lap)
    grid
    title('Laplacian Distribution')
    subplot(2,2,4)
    qqplot(Z)
    grid
    title('Uniform Distribution')
end