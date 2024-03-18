function [Y, X, index_ImMatch, matchedNumb, pointsPreviousNumb, pointsNumb, inlierNumb, status, error, time, matchedPoints, matchedPointsPrev, pointsPrevious, points, inlierPoints, inlierPointsPrev] = compute_trasnform(feature_function, I1, I2, index_matrix)
t  = tic;
pointsPrevious = feval(feature_function,I1);  %detectHarrisFeatures(I1); %detectBRISKFeatures(I1);  %detectMinEigenFeatures(I1);   %detectMSERFeatures(I1); %detectMSERFeatures(I1); %detectORBFeatures(I1); %detectKAZEFeatures(I1); %detectSIFTFeatures(I1);
points = feval(feature_function,I2);   %detectHarrisFeatures(I2); %detectBRISKFeatures(I2);  %detectMinEigenFeatures(I2);   %detectMSERFeatures(I2); %detectMSERFeatures(I2); %detectORBFeatures(I2); %detectKAZEFeatures(I2); %detectSIFTFeatures(I2);

[featuresPrevious, pointsPrevious] = extractFeatures(I1, pointsPrevious,"Upright",true);
[features, points] = extractFeatures(I2, points,"Upright",true);

pointsPreviousNumb = length(pointsPrevious);
pointsNumb = length(points);
if pointsNumb==0 || pointsPreviousNumb==0
    indexPairs=uint32(zeros(0,2));
else
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true,'Method','Exhaustive');%,'MatchThreshold',1,'MaxRatio',0.1 );
end
matchedPoints = points(indexPairs(:,1), :);
matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
matchedNumb = length(matchedPoints);

[tforms,inlierIndex, status] =  estimateGeometricTransform2D_customized(matchedPointsPrev, matchedPoints,'rigid', 'Confidence', 99.99, 'MaxNumTrials', 2000);
inlierPoints = matchedPoints(inlierIndex);
inlierPointsPrev = matchedPointsPrev(inlierIndex);

if status == 0
    index_ImMatch = index_matrix;
    inlierNumb = sum(inlierIndex);
    X = (tforms.Translation(1));
    Y = (tforms.Translation(2));
    e =((inlierPoints.Location-inlierPointsPrev.Location)-tforms.Translation).^2;
    error = sqrt(sum(e(:))/sum(inlierIndex));
else
    inlierPoints = NaN;
    inlierPointsPrev = NaN;
    index_ImMatch = NaN;
    X = NaN;
    Y = NaN;
    error = NaN;
    inlierNumb = sum(inlierIndex);
end
time = toc(t);
end