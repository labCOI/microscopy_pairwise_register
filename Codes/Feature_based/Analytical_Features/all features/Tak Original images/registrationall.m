function mainn = registrationall(location, feature_type, nb_horz_tiles, nb_vert_tiles, End, f_type, d_type)

feature_function = "detect"+feature_type+"Features";

t1 = tic;

source_directory = location;
files = dir(fullfile(source_directory,f_type));
files = natsortfiles({files.name});
Start = 1;
hop = 1;
%End = 100;     %% number of tiles
% Tak dataset 100/ human colon datset 609/ stem cell colony level1-2 100   level3 552/  small-phase or flourecent 25

file = files(Start:hop:End);
mainn.num_Im = numel(file);

index_matrix = 1:nb_vert_tiles*nb_horz_tiles;
t = nb_vert_tiles;
nb_vert_tiles = nb_horz_tiles;
nb_horz_tiles = t;

if d_type == 2
    % % for human colon dataset
    img_name_grid = reshape(file, nb_vert_tiles, nb_horz_tiles)';
    index_matrix = (reshape(index_matrix, nb_vert_tiles, nb_horz_tiles))';

    img_name_grid = img_name_grid(sort(1:size(img_name_grid ,1),'descend'),:);
    index_matrix = index_matrix(sort(1:size(index_matrix,1),'descend'),:);
else
    % % for Tak & stem cell colony dataset
    img_name_grid = reshape(file, nb_vert_tiles, nb_horz_tiles)';
    index_matrix = (reshape(index_matrix, nb_vert_tiles, nb_horz_tiles))';
end

[nb_vert_tiles, nb_horz_tiles] = size(img_name_grid);

mainn.Ty_north = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.Tx_north = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.matchedNumb_north = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.Ty_west= NaN(nb_vert_tiles,nb_horz_tiles);
mainn.Tx_west= NaN(nb_vert_tiles,nb_horz_tiles);

mainn.matchedNumb_west = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.error_north = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.error_west = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.index_ImMatch_north = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.pointsPreviousNumb_north = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.pointsNumb_north = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.index_ImMatch_west = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.pointsPreviousNumb_west = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.pointsNumb_west = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.inliersNumb_north = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.inliersNumb_west = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.time_north  = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.time_west  = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.RMSE_north  = NaN(nb_vert_tiles,nb_horz_tiles);
mainn.RMSE_west  = NaN(nb_vert_tiles,nb_horz_tiles);

mainn.matchedPoints_west = {};
mainn.matchedPointsPrev_west = {};
mainn.pointsPrevious_west = {};
mainn.points_west = {};
mainn.inlierPoints_west = {};
mainn.inlierPointsPrev_west = {};
mainn.matchedPoints_north = {};
mainn.matchedPointsPrev_north = {};
mainn.pointsPrevious_north = {};
mainn.points_north = {};
mainn.inlierPoints_north = {};
mainn.inlierPointsPrev_north = {};

[~ , ~, channel] = size(imread([source_directory img_name_grid{1,1}]));

for j = 1:nb_horz_tiles
    for i = 1:nb_vert_tiles
        fprintf('.');
        % read image from disk
        if channel == 3
            I1 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j}])));
        else
            I1 = im2double(imread([source_directory img_name_grid{i,j}]));
        end
        if i > 1
            % compute translation north
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i-1,j}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i-1,j}]));
            end
            [mainn.Ty_north(i,j), mainn.Tx_north(i,j), mainn.index_ImMatch_north(i,j), mainn.matchedNumb_north(i,j),...
                mainn.pointsPreviousNumb_north(i,j), mainn.pointsNumb_north(i,j), mainn.inliersNumb_north(i,j),...
                mainn.status_north(i,j), mainn.error_north(i,j), mainn.time_north(i,j), mainn.matchedPoints_north{i,j},...
                mainn.matchedPointsPrev_north{i,j}, mainn.pointsPrevious_north{i,j}, mainn.points_north{i,j},...
                mainn.inlierPoints_north{i,j}, mainn.inlierPointsPrev_north{i,j}] ...
                = compute_trasnform(feature_function,I1, I2, index_matrix(i,j));
        end
        if j > 1
            % perform pciam west
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j-1}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i,j-1}]));
            end
            [mainn.Ty_west(i,j), mainn.Tx_west(i,j), mainn.index_ImMatch_west(i,j),mainn.matchedNumb_west(i,j),...
                mainn.pointsPreviousNumb_west(i,j), mainn.pointsNumb_west(i,j), mainn.inliersNumb_west(i,j),...
                mainn.status_west(i,j), mainn.error_west(i,j), mainn.time_west(i,j), mainn.matchedPoints_west{i,j},...
                mainn.matchedPointsPrev_west{i,j}, mainn.pointsPrevious_west{i,j}, mainn.points_west{i,j},...
                mainn.inlierPoints_west{i,j}, mainn.inlierPointsPrev_west{i,j}] ...
                = compute_trasnform(feature_function,I1, I2, index_matrix(i,j));
        end
    end
end

mainn.pairwise_time = toc (t1)
mainn.average_time = mean(cat(2,mean(mainn.time_north(2:end,:),'all'),mean(mainn.time_west(:,2:end),'all')));

for j = 1:nb_horz_tiles
    for i = 1:nb_vert_tiles
        %     read image from disk
        if channel == 3
            I1 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j}])));
        else
            I1 = im2double(imread([source_directory img_name_grid{i,j}]));
        end
        if i > 1
            %       compute RMSE  between overlaped region north
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i-1,j}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i-1,j}]));
            end
            if ~isnan( mainn.index_ImMatch_north(i,j))
                sub_I2 = extract_subregion(I2,  mainn.Tx_north(i,j),  mainn.Ty_north(i,j));
                sub_I1 = extract_subregion(I1, - mainn.Tx_north(i,j), - mainn.Ty_north(i,j));
                mainn.RMSE_north(i,j)  = sqrt(mean((sub_I2(:)-sub_I1(:)).^2));
            end
        end
        if j > 1
            %       compute RMSE  between overlaped region west
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j-1}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i,j-1}]));
            end
            if ~isnan( mainn.index_ImMatch_west(i,j))
                sub_I2 = extract_subregion(I2,  mainn.Tx_west(i,j),  mainn.Ty_west(i,j));
                sub_I1 = extract_subregion(I1, - mainn.Tx_west(i,j), - mainn.Ty_west(i,j));
                mainn.RMSE_west(i,j)  = sqrt(mean((sub_I2(:)-sub_I1(:)).^2));
            end
        end
    end
end
mainn.average_RMSE = mean(cat(2,mean(mainn.RMSE_north(~isnan(mainn.RMSE_north))),mean(mainn.RMSE_west(~isnan(mainn.RMSE_west)))));
end
