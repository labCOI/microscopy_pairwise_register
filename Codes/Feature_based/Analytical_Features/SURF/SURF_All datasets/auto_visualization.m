function auto_visualization(first_imagenumb,second_imagenumb,which_feature,which_mode,which_dataset)
if abs(first_imagenumb-second_imagenumb)==10 || abs(first_imagenumb-second_imagenumb)==29
    north_west="_north"; %_north , _west
elseif abs(first_imagenumb-second_imagenumb)==1
    north_west="_west"; %_north , _west
else
    fprintf("image number is incorrect");
end

if which_dataset=="T02"
    dataset=dataset_dir{1,:};
elseif which_dataset=="T05"
    dataset=dataset_dir{2,:};
elseif which_dataset=="T15"
    dataset=dataset_dir{3,:};
elseif which_dataset=="T19"
    dataset=dataset_dir{4,:};
elseif which_dataset=="T23"
    dataset=dataset_dir{5,:};
elseif which_dataset=="COLN"
    dataset=dataset_dir{6,:};
end
help_number_tak=(reshape(0:99,10,10))';
help_number_human=(reshape(1:609,29,21))';
help_number_takkk=(reshape(1:100,10,10))';

matchedPoint = sprintf('%s.%s_%s.matchedPoints%s',which_dataset, which_feature, which_mode,north_west);
matchedPointsPre = sprintf('%s.%s_%s.matchedPointsPrev%s',which_dataset, which_feature, which_mode,north_west);
RMSE=sprintf('%s.%s_%s.RMSE%s',which_dataset, which_feature, which_mode,north_west);
Time=sprintf('%s.%s_%s.time%s',which_dataset, which_feature, which_mode,north_west);
Error=sprintf('%s.%s_%s.error%s',which_dataset, which_feature, which_mode,north_west);
matchnumb=sprintf('%s.%s_%s.matchedNumb%s',which_dataset, which_feature, which_mode,north_west);



if which_dataset=="COLN"
    first_imagenumb=first_imagenumb;
    second_imagenumb=second_imagenumb;
    first_invert=help_number_human(first_imagenumb);
    second_invert=help_number_human(second_imagenumb);


    if north_west=="_north"
        % Concatenate the images vertically
        concatenated_image = [imread(sprintf('%sCOLNOR69MW2-cycle-1.ome_6_%03d.tif', dataset, second_imagenumb))...
            ;imread(sprintf('%sCOLNOR69MW2-cycle-1.ome_6_%03d.tif', dataset, first_imagenumb))];

        % Shift the second set of matched points vertically by the height of image1
        matchedPoints2_shifted = eval(sprintf('%s{second_invert}.Location', matchedPointsPre))...
            + [0, size(imread(sprintf('%sCOLNOR69MW2-cycle-1.ome_6_%03d.tif', dataset, second_invert)), 1)];

        % Display the matched features with lines
        figure;
        imshow(concatenated_image);
        hold on;
        plot(eval(sprintf('%s{second_invert}.Location(:, 1)', matchedPoint)),eval(sprintf('%s{second_invert}.Location(:, 2)', matchedPoint)), 'ro', 'MarkerSize', 5);
        plot(matchedPoints2_shifted(:, 1), matchedPoints2_shifted(:, 2),'g+', 'MarkerSize', 5);
        for i = 1:size(eval(sprintf('%s{first_invert}.Location(:, 1)', matchedPoint)), 1)
            line([eval(sprintf('%s{second_invert}.Location(i, 1)', matchedPoint)), matchedPoints2_shifted(i, 1)], ...
                [eval(sprintf('%s{second_invert}.Location(i, 2)', matchedPoint)), matchedPoints2_shifted(i, 2)], 'Color', 'y');
        end
        hold off;
        title(sprintf('%s - Between images %d and %d - %s Mode\nRMSE=%0.3f - time=%0.3f - matchpoints=%d - error=%0.3f', which_feature, first_imagenumb, second_imagenumb, which_mode...
            ,eval(sprintf('%s(%d)',RMSE,second_imagenumb)),eval(sprintf('%s(%0.1f)',Time,second_imagenumb)),eval(sprintf('%s(%d)',matchnumb,second_imagenumb))...
            ,eval(sprintf('%s(%d)',Error,second_imagenumb))));
    else

        showMatchedFeatures(imread(sprintf('%sCOLNOR69MW2-cycle-1.ome_6_%03d.tif', dataset, first_imagenumb))...
            ,imread(sprintf('%sCOLNOR69MW2-cycle-1.ome_6_%03d.tif', dataset, second_imagenumb))...
            ,eval(sprintf('%s{second_invert}', matchedPoint))...
            ,eval(sprintf('%s{second_invert}', matchedPointsPre)),"montage");
        title(sprintf('%s - Between images %d and %d - %s Mode\nRMSE=%0.3f - time=%0.3f - matchpoints=%d - error=%0.3f', which_feature, first_imagenumb, second_imagenumb, which_mode...
            ,eval(sprintf('%s(%d)',RMSE,second_invert)),eval(sprintf('%s(%0.1f)',Time,second_invert)),eval(sprintf('%s(%d)',matchnumb,second_invert))...
            ,eval(sprintf('%s(%d)',Error,second_invert))));
    end

else
    first_imagenumb=first_imagenumb+1;
    second_imagenumb=second_imagenumb+1;
    first_invert=help_number_takkk(first_imagenumb);
    second_invert=help_number_takkk(second_imagenumb);

    if north_west=="_north"

        % Concatenate the images vertically
        concatenated_image = [imread(sprintf('%s%04d.jpg', dataset, first_imagenumb-1)) ;imread(sprintf('%s%04d.jpg', dataset, second_imagenumb-1))];

        % Shift the second set of matched points vertically by the height of image1
        matchedPoints2_shifted = eval(sprintf('%s{second_invert}.Location', matchedPointsPre))...
            + [0, size(imread(sprintf('%s%04d.jpg', dataset, second_imagenumb-1)), 1)];

        % Display the matched features with lines
        figure;
        imshow(concatenated_image);
        hold on;
        plot(eval(sprintf('%s{second_invert}.Location(:, 1)', matchedPoint)),eval(sprintf('%s{second_invert}.Location(:, 2)', matchedPoint)), 'ro', 'MarkerSize', 5);
        plot(matchedPoints2_shifted(:, 1), matchedPoints2_shifted(:, 2),'g+', 'MarkerSize', 5);
        for i = 1:size(eval(sprintf('%s{second_invert}.Location(:, 1)', matchedPoint)), 1)
            line([eval(sprintf('%s{second_invert}.Location(i, 1)', matchedPoint)), matchedPoints2_shifted(i, 1)], ...
                [eval(sprintf('%s{second_invert}.Location(i, 2)', matchedPoint)), matchedPoints2_shifted(i, 2)], 'Color', 'y');
        end
        hold off;
        title(sprintf('%s - Between images %d and %d - %s Mode\nRMSE=%0.3f  - time=%0.3f - matchpoints=%d - error=%0.3f', which_feature, first_imagenumb-1, second_imagenumb-1, which_mode...
            ,eval(sprintf('%s(%d)',RMSE,second_invert)),eval(sprintf('%s(%d)',Time,second_invert)),eval(sprintf('%s(%d)',matchnumb,second_invert)) ...
            ,eval(sprintf('%s(%d)',Error,second_invert))));
    else

        showMatchedFeatures(imread(sprintf('%s%04d.jpg', dataset, first_imagenumb-1))...
            ,imread(sprintf('%s%04d.jpg', dataset, second_imagenumb-1))...
            ,eval(sprintf('%s{second_invert}', matchedPoint))...
            ,eval(sprintf('%s{second_invert}', matchedPointsPre)),"montage");
        title(sprintf('%s - Between images %d and %d - %s Mode\nRMSE=%0.3f - time=%0.3f - matchpoints=%d - error=%0.3f', which_feature, first_imagenumb-1, second_imagenumb-1, which_mode...
            ,eval(sprintf('%s(%d)',RMSE,second_invert)),eval(sprintf('%s(%0.1f)',Time,second_invert)),eval(sprintf('%s(%d)',matchnumb,second_invert))...
            ,eval(sprintf('%s(%d)',Error,second_invert))));
    end
end
end