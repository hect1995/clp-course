% prac2_fonemas_PCA.m
clear;
close all;  % close all previous figures

%% Options / Initalitation
i_dib=0;				 %0 NO /1 YES: plot spectrums
N_coor = 256;
V_coor=1:N_coor;             %64 to take all features set 1:64
% V_coor=[22 64];         % EXAMPLE: Selection of a subset of two features

N_feat=length(V_coor);
% class name: Labels:
% 1(aa);2(ao);3(dcl);4(iy);5(sh);
N_classes=5;
N_fft=256;						%256 (8KHz) 128 (4KHz), 64 (2KHz), 32(1khZ)
%% Database load
load BD_phoneme

%% MEAN IS REMOVED FROM DATABASE
X=X-ones(length(Labels),1)*mean(X);

%% Spectrum plot
if i_dib==1
    Frec_max=8*N_fft/256;			%Max frequency in KHz
    eje_frec=(0:N_fft-1)*Frec_max/N_fft;
    clases=['aa';'ao';'dc';'iy';'sh'];
    figure('name','LOG(Espectrum)')
    for i_clas=1:N_classes
        subplot(3,2,i_clas)
        hold on
        index=find(Labels==i_clas);
        for i1=1:length(index)
            plot(eje_frec,X(index(i1),1:N_fft));
        end
        hold off
        grid
        zoom on
        xlabel('frec(KHz)')
        ylabel(clases(i_clas,:));
    end
    subplot(3,2,N_classes+1)
    hold on
    i_color=['b' 'r' 'g' 'k' 'y'];
    for i_clas=1:N_classes
        index= Labels==i_clas;
        aux=mean(X(index,1:N_fft));
        plot(aux,i_color(i_clas));
    end
    hold off
    grid
    zoom on
    xlabel('Feature Number')
    ylabel('log espectro')
    title('Average');
    clear index aux i_color i_clas eje_frec Frec_max
end
% clear i_dib N_fft

%% Feature selection
if V_coor(1)~=0
    X=X(:,V_coor);  % Feature selection
end
% clear V_coor

%% Database partition
P_train=0.7;
Index_train=[];
Index_test=[];
for i_class=1:N_classes
    index=find(Labels==i_class);
    N_i_class=length(index);
    [I_train,I_test] = dividerand(N_i_class,P_train,1-P_train);
    Index_train=[Index_train;index(I_train)];
    Index_test=[Index_test;index(I_test)];
end
% Train Selection
X_train=X(Index_train,:);
Labels_train=Labels(Index_train);
% Test Selection and mixing
X_test=X(Index_test,:);
Labels_test=Labels(Index_test);
% clear Index_train Index_test index i_class N_i_class I_train I_test

%% Projections
W = pca(X_train);
%% Data projection
X_train_proj = X_train * W;
X_test_proj = X_test * W;

LC_train_Pe = zeros(N_coor,1);
LC_test_Pe = zeros(N_coor,1);
QC_train_Pe = zeros(N_coor,1);
QC_test_Pe = zeros(N_coor,1);

% TODO Select d columns, compute error probabilities and plot graphics
tic
parfor d=1:N_coor
    %% Create a default (linear) discriminant analysis classifier:
    linclass = fitcdiscr(X_train_proj(:,1:d),Labels_train,'prior','empirical')
    
    Linear_out = predict(linclass,X_train_proj(:,1:d));
    Linear_Pe_train=sum(Labels_train ~= Linear_out)/length(Labels_train);
    fprintf(1,' error Linear train = %g   \n', Linear_Pe_train)
    
    LC_train_Pe(d) = Linear_Pe_train;
    
    Linear_out = predict(linclass,X_test_proj(:,1:d));
    Linear_Pe_test=sum(Labels_test ~= Linear_out)/length(Labels_test);
    fprintf(1,' error Linear test = %g   \n', Linear_Pe_test)
    
    LC_test_Pe(d) = Linear_Pe_test;
    
    %% Create a quadratic discriminant analysis classifier:
    quaclass = fitcdiscr(X_train_proj(:,1:d),Labels_train,'discrimType','quadratic','prior','empirical')
    
    Quadratic_out= predict(quaclass,X_train_proj(:,1:d));
    Quadratic_Pe_train=sum(Labels_train ~= Quadratic_out)/length(Labels_train);
    fprintf(1,' error Quadratic train = %g   \n', Quadratic_Pe_train)
    
    QC_train_Pe(d) = Quadratic_Pe_train;
    
    Quadratic_out= predict(quaclass,X_test_proj(:,1:d));
    Quadratic_Pe_test=sum(Labels_test ~= Quadratic_out)/length(Labels_test);
    fprintf(1,' error Quadratic test = %g   \n', Quadratic_Pe_test)
    
    QC_test_Pe(d) = Quadratic_Pe_test;
    
    %% Test confusion matrices
    CM_Linear_test=confusionmat(Labels_test,Linear_out)
    CM_Quadratic_test=confusionmat(Labels_test,Quadratic_out)
    
    % Print d'
    d
end
toc

plot(LC_train_Pe); hold on
plot(LC_test_Pe);
plot(QC_train_Pe);
plot(QC_test_Pe);

title(sprintf('Probabilidades de error LC/QC para train y test para d'' = 1..%d', N_coor));
grid on
legend('LC train', 'LC test' , 'QC train', 'QC test', 'best');

hold off