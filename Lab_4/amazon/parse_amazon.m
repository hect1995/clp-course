% Parse Amazon users dataset
clear

%% Open file
am = fopen('Amazon_initial_50_30_10000.arff', 'r');

%% Read data into cell array
% The headerlines argument skips all lines that are not useful data
data = textscan(am,'%s','headerlines',10005);

% Close file
fclose(am);

% The original 'data' is a 1x1 cell with a 1500x1 cell array inside
data = data{1,1};

%% Parse data into useful matrix
% Split by commas
data_split = split(data, ','); % Result is 1500x10001 string array
data_size = size(data_split);

%% Convert user names into numbers
names = {'Agresti','Ashbacher','Auken','Blankenship','Brody','Brown',...
    'Bukowsky','CFH','Calvinnme','Chachra','Chandler','Chell',...
    'Cholette','Comdet','Corn','Cutey','Davisson','Dent','Engineer',...
    'Goonan','Grove','Harp','Hayes','Janson','Johnson','Koenig',...
    'Kolln','Lawyeraau','Lee','Lovitt','Mahlers2nd','Mark','McKee',...
    'Merritt','Messick','Mitchell','Morrison','Neal','Nigam',...
    'Peterson','Power','Riley','Robert','Shea','Sherwin','Taylor',...
    'Vernon','Vision','Walters','Wilson'};

for i=1:data_size(1)
    % Find the index in 'names' of the name at the end of the i-th vector
    [a,b]=find(data_split(i, end) == names); %b is the name's index
    
    % Assign the name's index into data_split instead of the name's string
    data_split(i, end) = b-1; % Index names starting from 0
end

%% Convert string array into numeric matrix
amazon_users = str2double(data_split);

save('amazon_users.mat', amazon_users)
