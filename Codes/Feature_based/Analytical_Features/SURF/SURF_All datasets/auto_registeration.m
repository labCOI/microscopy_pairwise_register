function auto_registeration(dataPath, p)

Feature_type = ["SURF"]; %["SIFT", "FAST", "SURF", "BRISK", "KAZE", "Harris", "MinEigen", "MSER", "ORB"];

if p == 11              %% for human colon dataset
    width = 29; height = 21; overlapp = 0.03; End = 609; f_type = '*.tif'; d_type = 2;

elseif p == 12            %% for stem cell colony (SCC) dataset: small_phase
    width = 5; height = 5; overlapp = 0.19; End = 25; f_type = '*.tif'; d_type = 4;

elseif p == 13   %% for stem cell colony (SCC) dataset: phase
    width = 10; height = 10; overlapp = 0.1; End = 100; f_type = '*.tif'; d_type = 4;

elseif p == 14   %% for stem cell colony (SCC) dataset: Level1
    width = 10; height = 10; overlapp = 0.1; End = 100; f_type = '*.tif'; d_type = 4;

elseif p == 15   %% for stem cell colony (SCC) dataset: small
    width = 5; height = 5; overlapp = 0.19; End = 25; f_type = '*.tif'; d_type = 3;

elseif p == 16   %% for stem cell colony (SCC) dataset: Level2
    width = 10; height = 10; overlapp = 0.1; End = 100; f_type = '*.tif'; d_type = 3;

elseif p == 17   %% for stem cell colony (SCC) dataset: Level3
    width = 23; height = 24; overlapp = 0.1; End = 552; f_type = '*.tif'; d_type = 3;

else            %% for Tak dataset
    width = 10; height = 10; overlapp = 0.25; End = 100; f_type = '*.jpg'; d_type = 1;
end

for ij = 1:length(Feature_type)
    eval(sprintf('%s_overlap = registrationoverlap(dataPath, Feature_type(ij), width, height, overlapp, overlapp, End, f_type, d_type);', Feature_type(ij)));
    fprintf('Overlap feature: %s ok \n',Feature_type(ij));
    eval(sprintf('%s_all = registrationall(dataPath, Feature_type(ij), width, height, End, f_type, d_type);', Feature_type(ij)));
    fprintf('All feature : %s ok \n',Feature_type(ij));
end

if p==1; save("T02"); elseif p==2; save("T05"); elseif p==3; save("T15"); elseif p==4; save("T19");
elseif p==5; save("T23"); elseif p==6; save("T31"); elseif p==7; save("T33"); elseif p==8; save("T36");
elseif p==9; save("T49"); elseif p==10; save("T53");
elseif p==11; save("COLN");
elseif p==12; save("small_phase"); elseif p==13; save("phase"); elseif p==14; save("Level1");
elseif p==15; save("small"); elseif p==16; save("Level2"); elseif p==17; save("Level3");
else
    save("NEW")
end
end