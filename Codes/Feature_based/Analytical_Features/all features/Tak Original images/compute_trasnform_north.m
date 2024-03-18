function [Y1, X1, index_ImMatch1, matchedNumb1, pointsPreviousNumb1, pointsNumb1, inlierNumb1, status, error, time, matchedPoints, matchedPointsPrev, pointsPrevious, points, inlierPoints, inlierPointsPrev] = compute_trasnform_north(feature_function, I1, I2, index_matrix, Y_pixel, OvY, M, N)
t2 = tic;

if feature_function=="detectSIFTFeatures"
    % I1=[   I1(1:(floor(size(I1,1)*OvY)), :);zeros(size(I1,1)-floor(size(I1,1)*OvY),(size(I1,2) )) ];
    % pointsPrevious = feval(feature_function,I1); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    I1_crop = [I1(1:(floor(size(I1,1)*OvY)), :)];
    pointsPrevious = feval(feature_function,I1_crop); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [featuresPrevious, pointsPrevious] = extractFeatures(I1_crop, pointsPrevious,'Upright',true);

    size_image2_before = size(I2,1);
    I2_crop =[I2((end+1-(floor(size(I2,1)*OvY))):end , :)];
    size_image2_after = size(I2_crop,1);
    points = feval(feature_function,I2_crop);
    [features, points] = extractFeatures(I2_crop, points,'Upright',true);
    points.Location(:,2) = points.Location(:,2)+(size_image2_before-size_image2_after);

    % I2=[zeros(size(I2,1)-floor(size(I2,1)*OvY),(size(I2,2) ));I2((end+1-(floor(size(I2,1)*OvY))):end , :)];
    % points = feval(feature_function,I2);
    % [features, points] = extractFeatures(I2, points,'Upright',true);
else
    pointsPrevious = feval(feature_function,I1, 'ROI',[1 1 N Y_pixel]);  %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [featuresPrevious, pointsPrevious] = extractFeatures(I1, pointsPrevious,'Upright',true);

    points = feval(feature_function,I2,'ROI',[1 round(M*(1-OvY)) N Y_pixel]); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [features, points] = extractFeatures(I2, points,'Upright',true);
end

pointsPreviousNumb1 = length(pointsPrevious);
pointsNumb1 = length(points);

if pointsNumb1==0 || pointsPreviousNumb1==0
    indexPairs=uint32(zeros(0,2));
else
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true,'Method','Exhaustive');%,'MatchThreshold',1,'MaxRatio',0.1 );
end
matchedPoints = points(indexPairs(:,1), :);

matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
matchedNumb1 = length(matchedPoints); %added matchedpoint instead of indexpair

[tforms1,inlierIndex, status] =  estimateGeometricTransform2D_customized(matchedPointsPrev, matchedPoints,'rigid', 'Confidence', 99.99, 'MaxNumTrials', 2000);
inlierPoints = matchedPoints(inlierIndex);
inlierPointsPrev = matchedPointsPrev(inlierIndex);

if status == 0
    index_ImMatch1 = index_matrix;
    inlierNumb1 = sum(inlierIndex);
    X1 = (tforms1.Translation(1));
    Y1 = (tforms1.Translation(2));
    e =((inlierPoints.Location-inlierPointsPrev.Location)-tforms1.Translation).^2;
    error = sqrt(sum(e(:))/sum(inlierIndex));
else
    inlierPoints = NaN;
    inlierPointsPrev = NaN;
    index_ImMatch1 = NaN;
    X1 = NaN;
    Y1 = NaN;
    error = NaN;
    inlierNumb1 = sum(inlierIndex);
end
time = toc(t2);
end
