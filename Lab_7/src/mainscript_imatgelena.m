%% Exercise 2.2 script of the K-Means classifier
clear

% Read Lena image
imageclp = imread('images/lena.jpg');

original_red = reshape(imageclp(:,:,1),1,[]);
original_green = reshape(imageclp(:,:,2),1,[]);
original_blue = reshape(imageclp(:,:,3),1,[]);
image_rgb = [original_red ; original_green ; original_blue ];

%% Section 1
% Requantify image with a k-means algorithm

d= 3; % RGB represents 3 dimensions
K= 7; % Compute 7 centroids

% 1st centroids column represents all dimensions of red's centroids
[Centroids_rgb, Labels_rgb, n_rgb , J_rgb, trace1_rgb, trace2_rgb, ...
    Sw_rgb(:,:,K-1), Sb_rgb(:,:,K-1)] = CLP_Kmeans(image_rgb(1:d, :),K,d);

% Reconstruct re-quantified image

% Take the centroid of the last iteration
Centroides_definitiu = Centroids_rgb(:,:,end);

% Transpose the matrix to adapt it to the shape of the centroids matrix
Labels_rgb = Labels_rgb';

% Pre-allocate the matrix of the result image
vector_image = zeros([1, size(image_rgb')]);

for i = 1:length(original_red)
    % Assign the corresponding centroid to each pixel, according to its label
    vector_image(1,i,:) = Centroides_definitiu(:,Labels_rgb(1,i)); 
end

% Reshape back into a 3-channel image
requantified_lena = uint8(reshape(vector_image, size(imageclp)));

%% Section 2
% Represent the separate components of the original and re-quantified image

% TODO Plot these images in subplots
% Original image
figure, imshow(imageclp),        title('Original image')
figure, imshow(imageclp(:,:,1)), title('Original Red channel')
figure, imshow(imageclp(:,:,2)), title('Original Green channel')
figure, imshow(imageclp(:,:,3)), title('Original Blue channel')

% Re-quantified image
figure, imshow(requantified_lena),        title('Re-quantified image')
figure, imshow(requantified_lena(:,:,1)), title('Re-quantified Red channel')
figure, imshow(requantified_lena(:,:,2)), title('Re-quantified Green channel')
figure, imshow(requantified_lena(:,:,3)), title('Re-quantified Blue channel')

% Show clusters of original image
% TODO plot these clusters in a subplot
figure
scatter3(original_red, original_green, original_blue, 10, Labels_rgb)
title('Clusters of original image')

% Show clusters of re-quantified image
figure
scatter3(vector_image(:,:,1), vector_image(:,:,2), vector_image(:,:,3), ...
    10, Labels_rgb)
title('Clusters of re-quantified image')

%% TODO: Finish Section 3
% Display the evolution of the cost function

figure, plot(1:length(J_rgb),J_rgb), hold on
title('Cost Function','FontSize',16)
xlabel('Iteration','FontSize',14)
ylabel('Cost','FontSize',14)
grid on, hold off

% TODO: queda la part de les funcions de matrius de traça

%% Section 4
% Evaluate how many bits we need to store the original and re-quantified images

numero_bits = 8 * (numel(imageclp));
disp(['We need ', num2str(numero_bits), ' bits to store the Lena image']);

K_quant = 16;
numero_bits_codificada = log2(K_quant) * (numel(imageclp));
disp(['We need ', num2str(numero_bits_codificada), ...
    ' bits to store the quantified image']);

%% TODO: Fix Section 5
% Import a different image and apply Lena's centroids

imageclp2 = imread('images/PacoLucia.jpg');

original_red_p = reshape(imageclp2(:,:,1),1,[]);
original_green_p = reshape(imageclp2(:,:,2),1,[]);
original_blue_p = reshape(imageclp2(:,:,3),1,[]);
image_rgbp = [original_red_p ; original_green_p ; original_blue_p ];

vector_image_paco = zeros([1, size(image_rgbp')]);

for i = 1:length(vector_image_paco)
    norms = sum(abs(repmat(...
        double(image_rgbp(:,i)), 1, K) - double(Centroides_definitiu)).^2,1);
    [Minim_value, index] = min(norms);
    
    % Assign the RGB value of the closest centroid
    vector_image_paco(1,i,:) = Centroides_definitiu(:,index);
end

requantified_paco = uint8(reshape(vector_image_paco, size(imageclp2)));

% TODO Plot these images in a subplot
figure
imshow(imageclp2)
title('Paco de Lucia''s original image','FontSize',16);

figure
imshow(requantified_paco)
title('Paco de Lucia''s Quantified image with K= 7','FontSize',16);