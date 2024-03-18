function [Y2, X2, index_ImMatch2, matchedNumb2, pointsPreviousNumb2, pointsNumb2, inlierNumb2, status,error, time, matchedPoints, matchedPointsPrev, pointsPrevious, points, inlierPoints, inlierPointsPrev] = compute_trasnform_west(feature_function, I1, I2, index_matrix, X_pixel, OvX, M, N)
t3  = tic;

if feature_function=="detectSIFTFeatures"
    % I1=[I1(:,1:(floor(size(I1,2)*OvX))),zeros(size(I1,1),(size(I1,2)-floor(size(I1,2)*OvX)) ) ];
    % pointsPrevious = feval(feature_function,I1); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    I1_crop = [I1(:,1:(floor(size(I1,2)*OvX)))];
    pointsPrevious = feval(feature_function,I1_crop); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [featuresPrevious, pointsPrevious] = extractFeatures(I1_crop, pointsPrevious,'Upright',true);

    size_image2_before=size(I2,2);
    I2_crop=[I2(:,end+1-(floor(size(I2,2)*OvX)):end)];
    size_image2_after=size(I2_crop,2);
    points = feval(feature_function,I2_crop); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [features, points] = extractFeatures(I2_crop, points,'Upright',true);
    points.Location(:,1)=points.Location(:,1)+(size_image2_before-size_image2_after);

    % I2=[zeros(size(I2,1),(size(I2,2)-floor(size(I2,2)*OvX)) ) , I2(:,end+1-(floor(size(I2,2)*OvX)):end)];
    % points = feval(feature_function,I2); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    % [features, points] = extractFeatures(I2, points,'Upright',true);
else
    pointsPrevious = feval(feature_function,I1, 'ROI',[1 1 X_pixel M]); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [featuresPrevious, pointsPrevious] = extractFeatures(I1, pointsPrevious,'Upright',true);

    points = feval(feature_function,I2, 'ROI',[round(N*(1-OvX)) 1 X_pixel M]); %detectHarrisFeatures(); %detectBRISKFeatures();  %detectMinEigenFeatures();   %detectMSERFeatures(); %detectMSERFeatures(); %detectORBFeatures(); %detectKAZEFeatures(); %detectSIFTFeatures();
    [features, points] = extractFeatures(I2, points,'Upright',true);
end
pointsPreviousNumb2 = length(pointsPrevious);
pointsNumb2 = length(points);
if pointsNumb2==0 || pointsPreviousNumb2==0
    indexPairs=uint32(zeros(0,2));
else
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true,'Method','Exhaustive');%,'MatchThreshold',1,'MaxRatio',0.1 );
end
matchedPoints = points(indexPairs(:,1), :);
matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
matchedNumb2 = length(matchedPoints);

[tforms2,inlierIndex, status] =  estimateGeometricTransform2D_customized(matchedPointsPrev, matchedPoints,'rigid', 'Confidence', 99.99, 'MaxNumTrials', 2000);
if status == 0
    inlierPoints = matchedPoints(inlierIndex);
    inlierPointsPrev = matchedPointsPrev(inlierIndex);
    index_ImMatch2 = index_matrix;
    inlierNumb2 = sum(inlierIndex);
    X2 = (tforms2.Translation(1));
    Y2 = (tforms2.Translation(2 ));
    e =((inlierPoints.Location-inlierPointsPrev.Location)-tforms2.Translation).^2;
    error = sqrt(sum(e(:))/sum(inlierIndex));
else
    inlierPoints = NaN;
    inlierPointsPrev = NaN;
    index_ImMatch2 = NaN;
    X2 = NaN;
    Y2 = NaN;
    error = NaN;
    inlierNumb2 = sum(inlierIndex);
end
time = toc(t3);
end