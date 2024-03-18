function [RMSE_north, RMSE_west, average_RMSE, time_north, time_west, average_time, Tx_north, Tx_west, Ty_north, Ty_west] = pairwise_registration_PhaseCorrelation_NCC(direc, imtype, nEnd, htile, vtile, dataset)

tic

source_directory = direc
if imtype == 1
    files= dir(fullfile(source_directory,'*.jpg'));
elseif imtype == 2
    files= dir(fullfile(source_directory,'*.tif'));
end
files = natsortfiles({files.name});
Start =1; 
hop = 1;
End = nEnd;    %% number of tiles    
% Tak dataset 100/ human colon datset 609/ stem cell colony level1-2 100   level3 552/  small-phase or flourecent 25

file = files(Start:hop:End);
num_Im = numel(file);

% Tak dataset 10*10 / human colon datset 29*21/ stem cell colony level1-2 10*10   level3 23*24/  small-phase or flourecent 5*5
nb_horz_tiles = htile;
nb_vert_tiles = vtile;

index_matrix = 1:nb_vert_tiles*nb_horz_tiles;
t = nb_vert_tiles;
nb_vert_tiles = nb_horz_tiles;
nb_horz_tiles = t;

if dataset == 1 || dataset == 2
    % % for Tak & stem cell colony dataset
    img_name_grid = reshape(file, nb_vert_tiles, nb_horz_tiles)';
    index_matrix = (reshape(index_matrix, nb_vert_tiles, nb_horz_tiles))';
elseif dataset == 3
    % for hunan colon dataset
    img_name_grid = reshape(file, nb_vert_tiles, nb_horz_tiles)';
    index_matrix = (reshape(index_matrix, nb_vert_tiles, nb_horz_tiles))';
    img_name_grid  = img_name_grid (sort(1:size(img_name_grid ,1),'descend'),:);
    index_matrix = index_matrix(sort(1:size(index_matrix,1),'descend'),:);
end

[nb_vert_tiles, nb_horz_tiles] = size(img_name_grid);

Ty_north = NaN(nb_vert_tiles,nb_horz_tiles);
Tx_north = NaN(nb_vert_tiles,nb_horz_tiles);
CC_north = NaN(nb_vert_tiles,nb_horz_tiles);
time_north = NaN(nb_vert_tiles,nb_horz_tiles);


Ty_west = NaN(nb_vert_tiles,nb_horz_tiles);
Tx_west = NaN(nb_vert_tiles,nb_horz_tiles);
CC_west = NaN(nb_vert_tiles,nb_horz_tiles);
time_west = NaN(nb_vert_tiles,nb_horz_tiles);

RMSE_north = NaN(nb_vert_tiles,nb_horz_tiles);
RMSE_west = NaN(nb_vert_tiles,nb_horz_tiles);

% %
NB_FFT_PEAKS = 2;
[M N channel] = size(imread([source_directory img_name_grid{1,1}]));


for j = 1:nb_horz_tiles
  print_to_command(['  col: ' num2str(j) '/' num2str(nb_horz_tiles)]);
    for i = 1:nb_vert_tiles
        %     read image from disk
        if channel == 3
            I1 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j}])));
        else
            I1 = im2double(imread([source_directory img_name_grid{i,j}]));
        end
            if i > 1
        %       compute pciam north
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i-1,j}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i-1,j}]));
            end
            [Ty_north(i,j), Tx_north(i,j), CC_north(i,j),time_north(i,j)] = pciam(I2, I1, StitchingConstants.NORTH,NB_FFT_PEAKS, 0);
        
            end
        if j > 1
            %       perform pciam west
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j-1}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i,j-1}]));
            end
            [Ty_west(i,j), Tx_west(i,j), CC_west(i,j),time_west(i,j)] = pciam(I2, I1, StitchingConstants.WEST, NB_FFT_PEAKS, 0);
        end
    end
end


pairwise_time = toc
average_time = mean(cat(2,mean(time_north(2:end,:),'all'),mean(time_west(:,2:end),'all')));

for j = 1:nb_horz_tiles
    fprintf('  col: %d  / %d\n', j ,nb_horz_tiles);
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
            
            sub_I2 = extract_subregion(I2, Tx_north(i,j), Ty_north(i,j));
            sub_I1 = extract_subregion(I1, -Tx_north(i,j), -Ty_north(i,j));
            RMSE_north(i,j)  = sqrt(mean((sub_I2(:)-sub_I1(:)).^2));
            
        end
        if j > 1
            %       compute RMSE  between overlaped region west
            
            if channel == 3
                I2 = im2double(rgb2gray(imread([source_directory img_name_grid{i,j-1}])));
            else
                I2 = im2double(imread([source_directory img_name_grid{i,j-1}]));
            end
            
            
            sub_I2 = extract_subregion(I2, Tx_west(i,j), Ty_west(i,j));
            sub_I1 = extract_subregion(I1, -Tx_west(i,j), -Ty_west(i,j));
            RMSE_west(i,j)  = sqrt(mean((sub_I2(:)-sub_I1(:)).^2));
            
        end
        
        
    end
end
average_RMSE = mean(cat(2,mean(RMSE_north(~isnan(RMSE_north))),mean(RMSE_west(~isnan(RMSE_west)))));
% save('workspace.mat')

end