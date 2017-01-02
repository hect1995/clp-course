%% CUANTIFICAR IMATGE
imageclp= imread('lena.jpg');
color1= reshape(imageclp(:,:,1),1,[]);
color2= reshape(imageclp(:,:,2),1,[]);
color3= reshape(imageclp(:,:,3),1,[]);
imatge_rgb= [color1 ; color2 ; color3 ];

d= 3; %3 colors RGB
K= 7; %obtindre 7 centroides
[Centroides_rgb, Labels_rgb, n_rgb , J_rgb, trace1_rgb, trace2_rgb, Sw_rgb(:,:,K-1), Sb_rgb(:,:,K-1)] = CLP_Kmeans_lena(imatge_rgb(1:d, :),K, d);
%la 1era columna es totes les dimensions del centroide del vermell

%% Apartat 3
figure;
plot(1:length(J_rgb),J_rgb);
title('Cost Function','FontSize',16);
xlabel('Iteration','FontSize',14);
ylabel('Cost','FontSize',14);
grid on;

numero_bits= (2^24)*length(imageclp(1,:,1))*length(imageclp(:,1,1));
text1= ['El numero de bits necesarios para guardar la imagen Lena es ',num2str(numero_bits)];
disp(text1);
K_codif= 16;
numero_bits_codificada= K_codif*3*length(imageclp(1,:,1))*length(imageclp(:,1,1));
text2= ['El numero de bits necesarios para guardar la imagen codificada es ',num2str(numero_bits_codificada)];
disp(text2);
%% SCATTER PLOT DE COM QUEDA L'ORIGINAL EN CLUSTERS
scatter3(color1,color2,color3,10,Labels_rgb)
%% QUEDA LO DE LES FUNCIONS DE MATRIUS DE TRAÇA


%% RECONSTRUCT IMAGE
Centroides_definitiu= Centroides_rgb(:,:,end); %em quedo amb el centroide de la ultima iteracio
Labels_rgb=Labels_rgb'; %ho transposo per adaptar les dades
vector_imatge= zeros(1,length(color1),3);%les dimensions de la imatge original pero en una vector
for i=1:length(color1)
    vector_imatge(1,i,:)= Centroides_definitiu(:,Labels_rgb(1,i)); %depenent del label de cada posicio li poso el seu centroide corresponent
end
imatge_definitiva= reshape(vector_imatge,[length(imageclp(:,1,1)),length(imageclp(1,:,1)),length(imageclp(1,1,:))]); %per tornar a convertir el vector en la forma de la imatge original
imatge_definitiva= uint8(imatge_definitiva);
image(imatge_definitiva)
%% Apartat 2
% Imatge original
red = imageclp(:,:,1); % Red channel
green = imageclp(:,:,2); % Green channel
blue = imageclp(:,:,3); % Blue channel
a = zeros(size(imageclp, 1), size(imageclp, 2));
just_red = cat(3, red, a, a);
just_green = cat(3, a, green, a);
just_blue = cat(3, a, a, blue);
figure, imshow(imageclp), title('Original image')
figure, imshow(just_red), title('Red channel')
figure, imshow(just_green), title('Green channel')
figure, imshow(just_blue), title('Blue channel')
% Imatge codificada
red = imatge_definitiva(:,:,1); % Red channel
green = imatge_definitiva(:,:,2); % Green channel
blue = imatge_definitiva(:,:,3); % Blue channel
a = zeros(size(imatge_definitiva, 1), size(imatge_definitiva, 2));
just_red = cat(3, red, a, a);
just_green = cat(3, a, green, a);
just_blue = cat(3, a, a, blue);
figure, imshow(imatge_definitiva), title('Original image')
figure, imshow(just_red), title('Red channel')
figure, imshow(just_green), title('Green channel')
figure, imshow(just_blue), title('Blue channel')

%% 5. IMPORTAR UNA IMATGE DIFERENT I APLICAR-LI ELS CENTROIDES DE LA LENA --> Amb la de Paco dona horrible
imageclp2= imread('PacoLucia.jpg');
color1p= reshape(imageclp2(:,:,1),1,[]);
color2p= reshape(imageclp2(:,:,2),1,[]);
color3p= reshape(imageclp2(:,:,3),1,[]);
imatge_rgbp= [color1 ; color2 ; color3 ];
vector_image_paco=zeros(1,length(color1p),3); %per posa la imatge en un vector i en la 3a dimensio components RGB
for i = 1:length(vector_image_paco) 
    norms = sum(abs(repmat(double(imatge_rgbp(:,i)), 1, K) - double(Centroides_definitiu)).^2,1);
    [Minim_value, index] = min(norms);
    vector_image_paco(1,i,:)= Centroides_definitiu(:,index); % li poso el RGB del centroide més proxim al valor real
end
imatge_definitivap= reshape(vector_image_paco,[length(imageclp2(:,1,1)),length(imageclp2(1,:,1)),length(imageclp2(1,1,:))]);
imatge_definitivap= uint8(imatge_definitivap);
figure;
image(imatge_definitivap)
title('Imagen Quantificada con K= 7','FontSize',16);