%% CUANTIFICAR IMATGE
imageclp= imread('lena.jpg');
color1= reshape(imageclp(:,:,1),1,[]);
color2= reshape(imageclp(:,:,2),1,[]);
color3= reshape(imageclp(:,:,3),1,[]);
imatge_rgb= [color1 ; color2 ; color3 ];

d= 3;
K= 7;
[Centroides_rgb, Labels_rgb, n_rgb , J_rgb, trace1_rgb, trace2_rgb, Sw_rgb(:,:,K-1), Sb_rgb(:,:,K-1)] = CLP_Kmeans(imatge_rgb(1:d, :),K, d);
%la 1era columna es totes les dimensions del centroide del vermell

figure;
plot(1:length(J_rgb),J_rgb);
title('Cost Function','FontSize',16);
xlabel('Iteration','FontSize',14);
ylabel('Cost','FontSize',14);
grid on;
%end
%% RECONSTRUCT IMAGE
Centroides_definitiu= Centroides_rgb(:,:,end);
Labels_rgb=Labels_rgb';
vector_imatge= zeros(1,length(color1),3);
for i=1:length(color1)
    vector_imatge(1,i,:)= Centroides_definitiu(:,Labels_rgb(1,i)); %depenent del label de cada posicio li poso el seu centroide corresponent
end
imatge_definitiva= reshape(vector_imatge,[length(imageclp(:,1,1)),length(imatgeclp(1,:,1)),length(imatgeclp(1,1,:))]); %per tornar a convertir el vector en la forma de la imatge original
imatge_definitiva= uint8(imatge_definitiva);
image(imatge_definitiva)