%% SCAPE pre-processing code

%Elizabeth Hillman, 8/2021,
%with revisions by 
%Patricia Cooney 1/2022, 
%Wenze Li - 04/2022,
%PC - 05/2022
%update - 10/2022

%this code processes 3D tiffs for R and G channels of SCAPE data

%loads in pre-allocated frames of interest for each video
%loads tiffs of interest
%smooths voxels, bkgd subtract
%choose top-view or side-view
%calculates 3D avg for G & R
%also makes a ratio image from 3D avgs
%saves outputs as tiffs for green and red 16-bit (2D projections of 3D averaged data)
%save output as mat for ratio (floating point numbers)

%%
%load in the excel with frame numbers of interest (col1-name, col2-start,
%col3-stop)

%eois = readtable('framesofinterest-decemberdualcolor_update.xlsx');

%choose top or sideview
%perspective = input('Top or Side? (top = 1, side = 2)');
perspective = 1;

%loop all vids of interest within dir
% for la = 1:height(eois)
%     larvaname = eois.Var1{la};
%     frames = eois.Var2(la):eois.Var3(la); 
% end

fr =  [28,62,nan,nan; 16,36,400,424; 79,129,nan,nan]; %[283,330; 1,512;

%run in batch:
larvanames = {'larva4_run2_eL2','larva10_run1_L2','larva10_run4_L2'}; %'larva2_run2','larva3_run1',
for l = 1:length(larvanames)
    clear ratio
    larvaname = larvanames{l};

    cd 'C:\Users\coone\Desktop\Patricia\newdataset-3D';
    larvafolder = [pwd,'\',larvaname];
    larvaf = dir([larvafolder '/**/','*allsecs']);
    cd(strcat(larvafolder,'\',larvaf.name));

    %troubleshooting options for 3D-Gaussian smoothing of orig data (sf), pixel fluor
    %threshold before including in average (pt), and window size (w)
%     sf = 5;
%     pt = 0.3;
%     w = 15; %15 gives ~whole muscle depth
    sf = [3, 7];
    filler = [85, 90, 95];
    puncta = [1200, 1350, 1500];%1400
    perccut = [0.3];%0.3
    %pt = 0.2;%[0.2, 0.3, 0.4, 0.5];
    %w = 10;%[2, 5, 8, 10, 12, 15];
    %shift = 3;
    frames = [fr(l,1):fr(l,2)];
    if l == 2
        frames = [frames, fr(l,3):fr(l,4)];
    end

    for s = 1:length(sf)
        for pu = 1:length(puncta)
            for pe = 1:length(perccut)
                for fi = 1:length(filler)
                for j = 1:length(frames)
                    fnameG = strcat('G_',larvaname,'_',num2str(frames(j)),'.tiff');
                    fnameR = strcat('R_',larvaname,'_',num2str(frames(j)),'.tiff');
                    Gdata = tiffLoad(fnameG);
                    Rdata = tiffLoad(fnameR);

                    if perspective == 1 %top-down view
                        GDatap = permute(Gdata(:,:,1:end-1),[1, 3, 2]);
                        RDatap = permute(Rdata(:,:,1:end-1),[1, 3, 2]);
                    elseif perspective == 2
                        %keep same dims - sideview
                    end

                    ss = size(RDatap);

                    % smoothing
                    GDatasm = smooth3(GDatap,'gaussian',[sf(s),sf(s),sf(s)]);
                    RDatasm = smooth3(RDatap,'gaussian',[sf(s),sf(s),sf(s)]);

                    % finding a threshold value over which to keep data - better than
                    % processing loads of zeros for the more complicated maximum calcs
                    tmpR = max(RDatasm,[],1);
                    [hR, vR] = hist(tmpR(:),100);
                    threshR2 = vR(1); %takes the max value across pixels and only look at what's above the lowest 2%

                    keepR = find(RDatasm>threshR2);
                    tmpR = uint8(zeros(size(RDatasm)));
                    tmpR(keepR)= 1; % 3D mask of 1's above threshold

                    keepyR = find(max(tmpR,[],2)>0); % convert to 2D mask

                    % reshape smoothed data for faster processing
                    GDatasmL = reshape(permute(GDatasm,[1 3 2]),[ss(1)*ss(3),ss(2)]);
                    RDatasmL = reshape(permute(RDatasm,[1 3 2]),[ss(1)*ss(3),ss(2)]);

                    %reshape the raw data for the actual averages
                    GDatapL = reshape(permute(GDatap,[1 3 2]),[ss(1)*ss(3),ss(2)]);
                    RDatapL = reshape(permute(RDatap,[1 3 2]),[ss(1)*ss(3),ss(2)]);

                    % prepare variables
%                     frontsurface = zeros([ss(1),ss(3)]);
%                     backsurface = zeros([ss(1),ss(3)]);
%                     thickness = zeros([ss(1),ss(3)]);
                    gcampave2_single = zeros([ss(1),ss(3)]);
                    redave2_single = zeros([ss(1),ss(3)]);
                    gcampave2_inside = zeros([ss(1),ss(3)]);
                    redave2_inside = zeros([ss(1),ss(3)]);
                    gcampave2_outside = zeros([ss(1),ss(3)]);
                    redave2_outside = zeros([ss(1),ss(3)]);
                    maxpos = zeros([ss(1),ss(3)]);
                    L = size(RDatasm,2);
                    toolow = 0;
                    %pixthresh = pt(pth); % average adjacent pixels that are > % of the max.
                    %win = w(wi); % using a window of +/- w around the max (within which we take the top 50% pixels)

                    %loops through the super-threshold pixels and makes new 2D
                    %images with dynamic z-window size
                    for i = 1:length(keepyR)
                        %find where the slope changes to plateau (guts of larva)
                        smtrace = smooth(RDatasmL(keepyR(i),:),7);
                        if smtrace(:) > puncta(pu)
                            puncind = find(smtrace>puncta(pu));
                            samptrace = smtrace(smtrace>filler(fi))<puncta(pu);
                            newpunc = mean(samptrace);
                            newtrace = [samptrace(samptrace(1:puncind(1)-1)), repmat(newpunc,[1,puncind]), samptrace(puncind(end)+1:end)];
                        else
                            newtrace = smtrace(smtrace>filler(fi));
                        end
                        
                        if length(newtrace)<5
                            newtrace = smtrace;
                            toolow = 1;
                        end
                        [maxpx, maxpos] = max(newtrace);
                        perc = sort(newtrace(find(newtrace>filler(fi))));
                        bottom = mean(perc(1:round(perccut(pe)*length(perc))));
                        thresh = bottom + perccut(pe)*(maxpx - bottom);
                        
                        %search from the max index -- where is the first
                        %point left and right sides below threshold; store
                        %index before. edge cases: if left hits 1, use 1
                        innerpx = [];
                        innermin = maxpos - 1;
                        if maxpos <= 2
                            innermin = 1;
                        else
                            while isempty(innerpx) && innermin > 0
                                if smtrace(innermin) < thresh && maxpos > 2 %tip pt
                                    innerpx = 1;
                                    innermin = innermin + 1;                         
                                else %keep searching left
                                    innermin = innermin - 1;
                                end
                            end
                        end
%                         %rpt for right
%                         outerpx = [];
%                         outermin = maxpos + 1;
%                         if maxpos >= 138
%                             outermin = 138;
%                         else
%                             while isempty(outerpx) && outermin > 1
%                                 if smtrace(outermin) < thresh && maxpos > 2 %tip pt
%                                     outerpx = 1;
%                                     outermin = outermin + 1;
%                                 else %keep searching right
%                                     outermin = outermin - 1;
%                                 end
%                             end
%                         end
%                         innermin = find(smtrace > thresh,10,'first');
%                         innermin = innermin(find(innermin > 8, 1, 'first'));
%                         outermin = find(smtrace > thresh,1,'last');
%                         outermin = outermin(find(outermin < 138,1,'first'));
                        
                        %innerinds = [find(smtrace>thresh,20,'first') ];
                        %innermin = innerinds(find(innerinds>8,1,'first'));
                        %split for 2 potential layers
                        winnyi = round(innermin:maxpos-(0.2*(maxpos-innermin)));
                        winnyi = winnyi(winnyi>0);
                        %round(maxpos-(0.6*maxpos-innermin):maxpos-(0.2*(maxpos-innermin)));
%                         winnyo = round(maxpos+(0.2*(outermin-maxpos)):outermin);
%                         winnyo = winnyo(winnyo<138);
%                         %winnyo = round(maxpos+(0.2*outermin-maxpos):maxpos+(0.6*(outermin-maxpos)));
%                         winnyall = min(winnyi):max(winnyo);
                        %winnyall = round(maxpos-(0.4*maxpos-innermin):maxpos+(0.4*(outermin-maxpos)));
                        
%                         use2 = winnyi-1+find(RDatasmL(keepyR(i),winnyi:winnyo)-innermin>pixthresh*p); %find FWHM from innermin to max
%                         thickness(keepyR(i)) = length(use2);
%                         frontsurface(keepyR(i)) = winnyi;
%                         backsurface(keepyR(i)) = winnyo;
                        if toolow == 1
%                             gcampave2_single(keepyR(i)) = filler; %here, taking mean across x and z values
%                             redave2_single(keepyR(i)) = filler;

                            gcampave2_inside(keepyR(i)) = filler(fi); %here, taking mean across x and z values
                            redave2_inside(keepyR(i)) = filler(fi);
% 
%                             gcampave2_outside(keepyR(i)) = filler; %here, taking mean across x and z values
%                             redave2_outside(keepyR(i)) = filler;
%                             
                            toolow = 0;
                        else
%                             gcampave2_single(keepyR(i)) = mean(GDatapL(keepyR(i),winnyall)); %here, taking mean across x and z values
%                             redave2_single(keepyR(i)) = mean(RDatapL(keepyR(i),winnyall));

                            gcampave2_inside(keepyR(i)) = mean(GDatapL(keepyR(i),winnyi)); %here, taking mean across x and z values
                            redave2_inside(keepyR(i)) = mean(RDatapL(keepyR(i),winnyi));
% 
%                             gcampave2_outside(keepyR(i)) = mean(GDatapL(keepyR(i),winnyo)); %here, taking mean across x and z values
%                             redave2_outside(keepyR(i)) = mean(RDatapL(keepyR(i),winnyo));
                        end
%                         
%                         figure
%                         plot(RDatasmL(keepyR(i),:));
%                         hold on
%                         scatter(innermin,RDatasmL(keepyR(i),innermin));
%                         hold on
%                         scatter(winnys,p*pixthresh)
%                         hold on
%                         scatter(winnye,p*pixthresh)
%                         hold on
%                         patch([winnys, winnye, winnye, winnys],[p*pixthresh, p*pixthresh,max(RDatasmL(keepyR(i),:)),max(RDatasmL(keepyR(i),:))],'b','FaceAlpha',0.5)
                       
                    end
                %then test if window size and pixthresh need to change
                %esp check if window size change for diff muscle depths:
                    %could for each frame, take some portion of the distribution from
                    %real min (outer edge of muscle) to max to inner min (inner edge of
                    %muscles)...

                    %to test if do this by window size explicit or by trying to see if
                    %bimodal distribution for layers of muscles, find what x,y px val
                    %is corresponding with Lts and overlap, then see what z distrib of
                    %fluorescence is


                    pause(0.1);
%                     figure
%                     subplot(1,2,1)
%                     imagesc(mat2gray(gcampave2))
%                     subplot(1,2,2)
%                     imagesc(mat2gray(redave2))

                    %% resize and rotate images so larva is proportional
                    if perspective == 1
                        imsz = size(gcampave2_single);
%                         gcresz = imresize(gcampave2_single,[imsz(1)*1.0999,imsz(2)*4]);
%                         gcreszrot_single = rot90(gcresz,-1);
                        
                        gcresz = imresize(gcampave2_inside,[imsz(1)*1.0999,imsz(2)*4]);
                        gcreszrot_inside = rot90(gcresz,-1);
                        
%                         gcresz = imresize(gcampave2_outside,[imsz(1)*1.0999,imsz(2)*4]);
%                         gcreszrot_outside = rot90(gcresz,-1);
                        
                        
%                         rcresz = imresize(redave2_single,[imsz(1)*1.0999,imsz(2)*4]);
%                         rcreszrot_single = rot90(rcresz,-1);
%                         
                        rcresz = imresize(redave2_inside,[imsz(1)*1.0999,imsz(2)*4]);
                        rcreszrot_inside = rot90(rcresz,-1);
                        
%                         rcresz = imresize(redave2_outside,[imsz(1)*1.0999,imsz(2)*4]);
%                         rcreszrot_outside = rot90(rcresz,-1);
                    end

            %         ratio(:,:,j) = ((gcreszrot)-80./rcreszrot);

                    %%
                    %save gcamp depth-averaged 2D
                    if perspective == 1
%                         gcampsave = uint16(gcreszrot_single);
%                         filename = strcat('single_newmeth_','rem',num2str(puncta(pu)),'_bot',num2str(perccut(pe)),'filler_',num2str(filler),'_smooth',num2str(sf(s)),'_top_R_based_depth_gcamp_',larvaname);
%                         imwrite(gcampsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
%                         pause(0.1)
%                         
                        gcampsave = uint16(gcreszrot_inside);
                        filename = strcat('inside_newmeth','rem',num2str(puncta(pu)),'_bot',num2str(perccut(pe)),'filler_',num2str(filler(fi)),'_smooth',num2str(sf(s)),'_top_R_based_depth_gcamp_',larvaname);
                        imwrite(gcampsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
                        pause(0.1)
                        
%                         gcampsave = uint16(gcreszrot_outside);
%                         filename = strcat('outside_newmeth','rem',num2str(puncta(pu)),'_bot',num2str(perccut(pe)),'filler_',num2str(filler),'_smooth',num2str(sf(s)),'_top_R_based_depth_gcamp_',larvaname);
%                         imwrite(gcampsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
%                         pause(0.1)
%                         
                        %save mcherry depth-averaged 2D
%                         redsave = uint16(rcreszrot_single);
%                         filename = strcat('single_newmeth','rem',num2str(puncta(pu)),'_bot',num2str(perccut(pe)),'filler_',num2str(filler),'_smooth',num2str(sf(s)),'_top_R_based_depth_mcherry_',larvaname);
%                         imwrite(redsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
                        
                        redsave = uint16(rcreszrot_inside);
                        filename = strcat('inside_newmeth','rem',num2str(puncta(pu)),'_bot',num2str(perccut(pe)),'filler_',num2str(filler(fi)),'_smooth',num2str(sf(s)),'_top_R_based_depth_mcherry_',larvaname);
                        imwrite(redsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
%                         
%                         redsave = uint16(rcreszrot_outside);
%                         filename = strcat('outside_newmeth','rem',num2str(puncta(pu)),'_bot',num2str(perccut(pe)),'filler_',num2str(filler),'_smooth',num2str(sf(s)),'_top_R_based_depth_mcherry_',larvaname);
%                         imwrite(redsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
                    elseif perspective == 2
            %             gcampsave = uint16(gcampave2);
            %             filename = strcat('side_fullrun_resize_rotate_pxthresh',num2str(pt),'_smooth',num2str(sf),'_win',num2str(w),'_imageunsmooth_top_R_based_depth_gcamp_',larvaname);
            %             imwrite(gcampsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
            %             %save mcherry depth-averaged 2D
            %             redsave = uint16(redave2);
            %             filename = strcat('side_fullrun_resize_rotate_pxthresh',num2str(pt),'_smooth',num2str(sf),'_win',num2str(w),'_imageunsmooth_top_R_based_depth_mcherry_',larvaname);
            %             imwrite(redsave, strcat(filename, '.tiff'), 'Tiff', 'Compression', 'none', 'WriteMode', 'Append');
                    end

                end
            end
        end
        end
    end

%     %save ratio
%     ratio(ratio==Inf) = 0;
%     ratio(ratio==-Inf) = 0;
%     save(strcat('fullrun_',larvaname,'_ratio.mat'),'ratio','-v7.3');
end
