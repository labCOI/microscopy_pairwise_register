function auto_registeration(dataPath, p)

if p == 11      %% for human colon dataset
    Feature_type = ["SIFT", "FAST", "SURF", "BRISK", "KAZE", "Harris", "MinEigen", "MSER"];
    width = 29; height = 21; overlapp = 0.03; End = 609; f_type = '*.tif'; d_type = 2;
elseif p > 11   %% for stem cell colony (SCC) dataset
    Feature_type = ["SIFT", "FAST", "SURF", "BRISK", "KAZE", "Harris", "MinEigen", "MSER", "ORB"];
    width = 10; height = 10; overlapp = 0.1; End = 100; f_type = '*.jpg'; d_type = 3;
else            %% for Tak dataset
    Feature_type = ["SIFT", "FAST", "SURF", "BRISK", "KAZE" ,"Harris", "MinEigen", "MSER" , "ORB"];
    width = 10; height = 10; overlapp = 0.25; End = 100; f_type = '*.jpg'; d_type = 1;
end

for ij = 1:length(Feature_type)
    eval(sprintf('%s_overlap = registrationoverlap(dataPath, Feature_type(ij), width, height, overlapp, overlapp, End, f_type, d_type);', Feature_type(ij)));
    fprintf('Overlap feature: %s ok \n',Feature_type(ij));
    % eval(sprintf('%s_all = registrationall(dataPath, Feature_type(ij), width, height, End, f_type, d_type);', Feature_type(ij)));
    % fprintf('All feature : %s ok \n',Feature_type(ij));
end

if p==1; save("T02"); elseif p==2; save("T05"); elseif p==3; save("T15"); elseif p==4; save("T19");
elseif p==5; save("T23"); elseif p==6; save("T31"); elseif p==7; save("T33"); elseif p==8; save("T36");
elseif p==9; save("T49"); elseif p==10; save("T53");
elseif p==11; save("COLN") ;
else 
    save("NEW")
end
end