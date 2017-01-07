function [ database, labels ] = CLP_Parse_DB( file_path )
%PARSE_DB Summary of this function goes here
%   Detailed explanation goes here

f = fopen(file_path, 'r');

data = textscan(f,'%f,%f,%f,%f,%s');

fclose(f);

s_length = data{1,1};
s_width = data{1,2};
p_length = data{1,3};
p_width = data{1,4};

classes = data{1,5}; % Classes are returned but not used (unsupervised learning)
labels = zeros(size(classes));

names = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};

for i=1:length(classes)
    switch classes{i}
        case names{1}
            labels(i) = 1;
        case names{2}
            labels(i) = 2;
        case names{3}
            labels(i) = 3;
        otherwise
            error(['Unknown class: ',classes{i}])
    end
end

database = [s_length, s_width, p_length, p_width]';

end

