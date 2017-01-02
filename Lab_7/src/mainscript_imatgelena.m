% %% Exercise 2.2 script of the K-Means classifier
% clear
% 
% % Read Lena image
% imageclp = imread('images/lena.jpg');
% 
% original_red = reshape(imageclp(:,:,1),1,[]);
% original_green = reshape(imageclp(:,:,2),1,[]);
% original_blue = reshape(imageclp(:,:,3),1,[]);
% image_rgb = [original_red ; original_green ; original_blue ];
% 
% %% Section 1
% %Requantify image with k-means
% 
% d= 3; % RGB represents 3 dimensions
% K= 7; % Compute 7 centroids
% 
% % 1st centroids column represents all dimensions of red's centroids
% [Centroids_rgb, Labels_rgb, n_rgb , J_rgb, trace1_rgb, trace2_rgb, ...
%     Sw_rgb(:,:,K-1), Sb_rgb(:,:,K-1)] = CLP_Kmeans_lena(image_rgb(1:d, :),K,d);
% 
% % Reconstruct re-quantified image
% Centroides_definitiu = Centroids_rgb(:,:,end); %em quedo amb el centroide de la ultima iteracio
% Labels_rgb = Labels_rgb'; %ho transposo per adaptar les dades
% vector_image = zeros([1, size(image_rgb')]);%les dimensions de la imatge original pero en una vector
% 
% for i = 1:length(original_red)
%     vector_image(1,i,:) = Centroides_definitiu(:,Labels_rgb(1,i)); %depenent del label de cada posicio li poso el seu centroide corresponent
% end
% 
% % Reshape back into a 3-channel image
% requantified_lena = uint8(reshape(vector_image, size(imageclp)));
% 
% %% Section 2
% % Represent the separate components of the original and re-quantified image
% 
% % Original image
% figure, imshow(imageclp),        title('Original image')
% figure, imshow(imageclp(:,:,1)), title('Original Red channel')
% figure, imshow(imageclp(:,:,2)), title('Original Green channel')
% figure, imshow(imageclp(:,:,3)), title('Original Blue channel')
% 
% % Re-quantified image
% figure, imshow(requantified_lena),        title('Re-quantified image')
% figure, imshow(requantified_lena(:,:,1)), title('Re-quantified Red channel')
% figure, imshow(requantified_lena(:,:,2)), title('Re-quantified Green channel')
% figure, imshow(requantified_lena(:,:,3)), title('Re-quantified Blue channel')
% 
% % Show clusters of original image
% figure
% scatter3(original_red, original_green, original_blue, 10, Labels_rgb)
% title('Clusters of original image')
% 
% % Show clusters of re-quantified image
% figure
% scatter3(vector_image(:,:,1), vector_image(:,:,2), vector_image(:,:,3), 10, Labels_rgb)
% title('Clusters of re-quantified image')
% 
% %% Section 3
% % Display the evolution of the cost function
% 
% figure, plot(1:length(J_rgb),J_rgb), hold on
% title('Cost Function','FontSize',16)
% xlabel('Iteration','FontSize',14)
% ylabel('Cost','FontSize',14)
% grid on, hold off
% 
% % TODO: queda la part de les funcions de matrius de traça
% 
% %% Section 4
% % Evaluate how many bits we need to store the original and re-quantified images
% 
% numero_bits = 8 * (numel(imageclp));
% disp(['We need ', num2str(numero_bits), ' bits to store the Lena image']);
% 
% K_codif = 16;
% numero_bits_codificada = log2(K) * (numel(imageclp));
% disp(['We need ', num2str(numero_bits_codificada), ...
%     ' bits to store the quantified image']);

%% Section 5
% Import a different image and apply Lena's centroids

imageclp2 = imread('images/PacoLucia.jpg');

color1p = reshape(imageclp2(:,:,1),1,[]);
color2p = reshape(imageclp2(:,:,2),1,[]);
color3p = reshape(imageclp2(:,:,3),1,[]);
image_rgbp = [original_red ; original_green ; original_blue ];

vector_image_paco = zeros([1, size(image_rgbp')]);

for i = 1:length(vector_image_paco)
    norms = sum(abs(repmat(double(image_rgbp(:,i)), 1, K) - double(Centroides_definitiu)).^2,1);
    [Minim_value, index] = min(norms);
    vector_image_paco(1,i,:) = Centroides_definitiu(:,index); % li poso el RGB del centroide m�s proxim al valor real
end

requantified_paco = uint8(reshape(vector_image_paco, size(imageclp2)));

figure
image(requantified_paco)
title('Paco de Lucia''s Quantified image with K= 7','FontSize',16);
% The image of Paco de Lucia gives a horrible result