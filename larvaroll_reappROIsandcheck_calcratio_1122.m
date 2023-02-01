%% extra frames - draw ROIs from orig ROI set onto extra frames at beginning and end of roll

%steps:
% reads in 3D-avg tiff data -- full stack, just the rolling frames
% loads in ROIs and traces from previous drawing
% watch how ROIs apply to mcherry data, allow redraw frames, extract new
%mcherry and gcamp values
% includes skipping capabilities
% implements 90% threshold and removal of mcherry; NO interpolation
% calc ratio of ROI each channel, then take mean
% plots trace for gcamp and mcherry; NO smoothing -- without 0 baseline
% allows checking ROIs if data looks weird (sharp peaks/noise appearance)
% allows redraw if check showed errors
% allows save movie of trace + ROIs

%saves ROI positions, fluorescence values, and various metadata

%Patricia Cooney, 05/2022

%% Locate the tiffs of interest
newload = input('Load Data? yes + images = 1, no, just mats = 0');
if newload == 1
    clear all
    close all
    
    dirname = uigetdir;
    cd = dirname;
    filenames = dir([dirname '/**/','*.tif']);
    file = filenames(1);
    expername = extractBetween(convertCharsToStrings(file.name),"gcamp_",".tif");

    %% Load in the separate channels
    %load
    for fi = 1:length(filenames)
        if contains(filenames(fi).name,'inside') && contains(filenames(fi).name,'gcamp')
            tiffnamegreen = filenames(fi).name;
            newinfo = imfinfo(tiffnamegreen);
            frames = length(newinfo); 
            for a = 1 : frames
                i_tempgreen(:,:,a) = imread(tiffnamegreen, a);
                %i_tempgreen(:,:,a) = imread(tiffnamegreen, a)./40; %adjustment line for when gcamp is far brighter than mcherry (larva10_run1 = ./40; larva10_run4 = ./40)
            end
        elseif contains(filenames(fi).name,'inside') && contains(filenames(fi).name,'mcherry')
            tiffnamered = filenames(fi).name;
            newinfo = imfinfo(tiffnamered);
            frames = length(newinfo); 
            for a = 1 : frames
                i_tempred(:,:,a) = imread(tiffnamered, a);
                %i_tempred(:,:,a) = imread(tiffnamered, a)./50; %adjust divide by 50 for larva4 expression
            end
        end
    end
    
    %load those mat files and check if already done and in spreadsheet 
    matfilenames = dir([pwd,'/**/','*recalcreapp','*_avgdata.mat']);
    
    %load in the calculated ratio mat
%     filenameratio = dir([dirname '/**/','*ratio.mat']);
%     i_tempratio = load(filenameratio);
    
else
    %keep going with loaded tiff
    clear alldata
    clear avgdata
    close all
end

%% repeat this process for the main ROIs you've already drawn: subset muscs and seg grps
fignames = dir([pwd,'/**/','*_rawtraces_wzeros.fig']);%check for already done
figlist = 'fig';
for fn = 1:length(fignames)
    figlist = strcat(figlist,fignames(fn).name);
end

for m = 1 : length(matfilenames)
    musclename = matfilenames(m).name(13:17);
    
    if contains(figlist,string(musclename))
        %skip, already done
    else
        clear loaded
        loaded = load(matfilenames(m).name,'avgdata');
        rois = loaded.avgdata.newrois;
        rollframes = find(~cellfun(@isempty,rois));
        
        %% Segmentation, DF/F Calculations, and Plotting Pipeline
        imagec = [88, 487]; %(scale from [60,480] OR [10,2800] is nice for most videos to show contrast; troubleshoot as needed)
        
        newtraces = zeros(size(i_tempgreen,3),4);
        newrois = cell(1,size(i_tempgreen,3));
        %run local segmentation function initially
        fixframes = rollframes;
            %just reapply all the old ROIs and check trace before redrawing
            auto = 1;
            [newtraces,newrois] = runseg(musclename,i_tempred,i_tempgreen,rois,fixframes,imagec,newtraces,newrois,auto);
            %plot those vals
            plottraces(newtraces,musclename);
            
            %option to redraw
            sh = input('Want to check the ROIs? yes = 1, no = 0');
            close all
            while sh == 1
                %Go in and check ROI locations for when the data looks very noisy
                redoframes = input('Which frames need attention? Enter as array (e.g. [1:3, 5]');
                fixframes = redoframes;
                close all
                auto = 0;
                [newtraces,newrois] = runseg(musclename,i_tempred,i_tempgreen,rois,fixframes,imagec,newtraces,newrois,auto);
                close all
                
                %plot those vals
                plottraces(newtraces,musclename);
                
                sh = input('Want to check the ROIs? yes = 1, no = 0');
            end
            %option - make video?
%             option = input('make a movie of this ROI? 1 = yes');
%             if option == 1
%                 makemovie(newtraces,musclename,i_tempratio,newrois,imagec,bend,initiate,roll,stop)
%             end
    end
    %save your values
    if ~contains(figlist,string(musclename))
        close all
        disp('Ok, saving all')
        %avgdata.expername = expername;
        %avgdata.musclename = musclename;
        avgdata.newtraces = newtraces;
        avgdata.newrois = newrois;

        %keep for potential plotting later
        peaktimes = max(newtraces);
        avgdata.peaktimes = peaktimes;

        save(strcat('recalcreapp_',musclename,'_avgdata.mat'),'avgdata','-v7.3')
        disp('Done')
    end
end
%%
%%
%%
%% local functions:
function [newtraces,newrois] = runseg(musclename,i_tempred,i_tempgreen,rois,fixframes,imagec,newtraces,newrois,auto)

    %make new ROI array with duplicates from orig roll rois and add in new
    for ar = 1:length(rois)
        if ~isempty(rois{ar})
            newrois{ar} = rois{ar};
        end
    end
    
    for foi = 1:length(fixframes)
        fr = fixframes(foi);

        i_segred = i_tempred(:,:,fr);
        i_seggreen = i_tempgreen(:,:,fr);
        if auto == 0
            if ~isempty(newrois{fr})                
                imshow(int16(i_seggreen),imagec);
                title(strcat('Current Frame: ',num2str(fr),' musclename: ',musclename))
                vertsarray = newrois{fr};   
                fa = [(1:length(newrois{fr})) 1];
                p = patch('Faces',fa,'Vertices',vertsarray,'FaceColor','red','FaceAlpha',.2);
                shg
            elseif isempty(newrois{fr})
                if fr == 1
                    prevframe = fr;
                else
                    prevframe = fr - 1;
                    if ~isempty(newrois{prevframe})
                        imshow(int16(i_seggreen),imagec);
                        title(strcat('Previous Frame: ',num2str(fr),' musclename: ',musclename))
                        vertsarray = newrois{prevframe};
                        fa = [(1:length(newrois{fr})) 1];
                        p = patch('Faces',fa,'Vertices',vertsarray,'FaceColor','red','FaceAlpha',.2);
                        shg
                    elseif isempty(newrois{prevframe})
                        imshow(int16(i_seggreen),imagec);
                        title(strcat('Draw new: ',num2str(fr),' musclename: ',musclename))
                        vertsarray = [];
                        shg
                    end
                end
            end

            drawnow = input('Draw New ROI? enter = yes, 0 = no, 1 = keep old ROI'); %draw or skip
            if isempty(drawnow) || drawnow == 1
                if isempty(drawnow) && ~isempty(newrois{fr})
                    shg
                    vertsarray = cell2mat(newrois(fr));
                    newroi_object = drawpolygon('Position',vertsarray);
                    hold on 
                    input('Verify New ROI');
                    shg
                    newrois{fr} = newroi_object.Position;
                    vertsarray = cell2mat(newrois(fr));
                elseif isempty(drawnow) && ~isempty(newrois{prevframe})
                    shg
                    vertsarray = cell2mat(newrois(prevframe));
                    newroi_object = drawpolygon('Position',vertsarray);
                    input('Verify New ROI');
                    shg
                    newrois{fr} = newroi_object.Position;
                    vertsarray = cell2mat(newrois(fr));
                elseif isempty(drawnow)
                    shg
                    newroi_object = drawpolygon;
                    input('Verify New ROI');
                    shg
                    newrois{fr} = newroi_object.Position;
                    vertsarray = cell2mat(newrois(fr));
                elseif drawnow == 1
                    vertsarray = cell2mat(newrois(fr));
                end
                
                newmred = poly2mask(vertsarray(:,1),vertsarray(:,2),size(i_segred,1),size(i_segred,2));
                newmgreen = poly2mask(vertsarray(:,1),vertsarray(:,2),size(i_seggreen,1),size(i_seggreen,2));

                %remove mcherry puncta and calculate mean values
                %all channels
                withpunctared_90 = double(i_segred(newmred));
                withpunctagreen_90 = double(i_seggreen(newmgreen));
                [~,reminds] = rmoutliers(withpunctared_90,'percentiles',[0,90]); %remove any fluor val's that are stat outlier for ROI in each frame
                wopunctared_90 = withpunctared_90(~reminds);
                wopunctagreen_90 = withpunctagreen_90(~reminds);

                newtraces(fr,1) = nanmean(wopunctagreen_90);
                newtraces(fr,2) = nanmean(i_segred(newmred));
                newtraces(fr,3) = nanmean(wopunctared_90); 
                
                %calculate the ratio of the ROI, then take the mean
                roiratio = (wopunctagreen_90./wopunctared_90);
                newtraces(fr,4) = nanmean(roiratio);
                
                hold off
            elseif drawnow == 0
                %save 0's and move to next frame
                newtraces(fr,1) = 0;
                newtraces(fr,2) = 0;
                newtraces(fr,3) = 0; 
                newtraces(fr,4) = 0;
            end
        elseif auto == 1
            vertsarray = cell2mat(newrois(fr));          
            newmred = poly2mask(vertsarray(:,1),vertsarray(:,2),size(i_segred,1),size(i_segred,2));
            newmgreen = poly2mask(vertsarray(:,1),vertsarray(:,2),size(i_seggreen,1),size(i_seggreen,2));

            %remove mcherry puncta and calculate mean values
            %all channels
            withpunctared_90 = double(i_segred(newmred));
            withpunctagreen_90 = double(i_seggreen(newmgreen));
            [~,reminds] = rmoutliers(withpunctared_90,'percentiles',[0,90]); %remove any fluor val's that are stat outlier for ROI in each frame
            wopunctared_90 = withpunctared_90(~reminds);
            wopunctagreen_90 = withpunctagreen_90(~reminds);

            newtraces(fr,1) = nanmean(wopunctagreen_90);
            newtraces(fr,2) = nanmean(i_segred(newmred));
            newtraces(fr,3) = nanmean(wopunctared_90); 

            %calculate the ratio of the ROI, then take the mean
            roiratio = (wopunctagreen_90./wopunctared_90);
            newtraces(fr,4) = nanmean(roiratio);
        end
    end
end
   
%%
function plottraces(newtraces,musclename) 
%isolate just the data from when muscles in the FOV
if find(newtraces(:,1)<=0,1,'first') > find(newtraces(:,1)>0,1,'first')
    spaceind = find(newtraces(:,1)<=0,1,'first');
    nonzerotraces = reshape(newtraces(newtraces>0),[],4);
    nonzerotraces = [nonzerotraces(1:spaceind-1,1:4); zeros(5,4); nonzerotraces(spaceind:end,1:4)];
else
    nonzerotraces = reshape(newtraces(newtraces>0),[],4);
end

    figure
    colororder({'k','b'})

    yyaxis left
    ylim([0,max(max(newtraces))*1.1])
    plot(newtraces(:,1),'g');
    hold on
    plot(newtraces(:,3),'r');
    hold on
    ylabel('Raw Traces')

    yyaxis right
    ylim([0,max(newtraces(:,4))*1.1])
    plot(newtraces(:,4),'b');
    hold on
    ylabel('Ratio Raw')

    xlim([1 length(newtraces)]);
    xlabel('Frames')
    title(strcat(musclename,' Roll Activity'))

    saveas(gcf,strcat(musclename,'_rawtraces_wzeros'),'jpeg')
    saveas(gcf,strcat(musclename,'_rawtraces_wzeros','.fig'))

    %only non-zero vals
    figure
    colororder({'k','b'})
    
    yyaxis left
    ylim([min(min(nonzerotraces)),max(max(nonzerotraces))*1.1])
    plot(nonzerotraces(:,1),'g');
    hold on
    plot(nonzerotraces(:,3),'r');
    hold on
    ylabel('Raw Traces')

    yyaxis right
    ylim([0,max(nonzerotraces(:,4))*1.1])
    plot(nonzerotraces(:,4),'b');
    hold on
    ylabel('Ratio Raw')

    xlim([1 length(nonzerotraces)]);
    xlabel('Frames')
    title(strcat(musclename,' - Roll Activity in FOV'))

    saveas(gcf,strcat(musclename,'_rawtraces_nonzeros'),'jpeg')
    saveas(gcf,strcat(musclename,'_rawtraces_nonzeros','.fig'))
end

%%
% function makemovie(newtraces,musclename,i_tempratio,newrois)
%     nonzerotraces = reshape(newtraces(traces>0),[],4);
%     
%     for fr = frames
%         fig = figure;
%         raw_avi(:,:,fr) = i_tempratio(:,:,fr);
%         vertsarray = newrois{fr};
%         %make subplot for SCAPE data
%         subplot(4,1,1:3);
%         title(strcat(musclename,' frame: ', num2str(fr)))
%         if newtraces(fr,:) > 0
%             imshow(int16(raw_avi(:,:,fr)),imagec);
%             hold on
%             %colored ROI based on saved verts
%             do = polyshape(vertsarray); 
%             h = plot(do);
%             h.EdgeColor = 'magenta';
%         elseif newtraces(fr,:) == 0
%             imshow(int16(raw_avi(:,:,fr)),imagec);
%             hold on  
%         end
%         %make subplot for trace data; add vertical bar for each frame value
%         subplot(4,1,4);
%         colororder({'k','b'})
%         yyaxis left
%         ylim([0,max(max(newtraces)*1.1)])
%         ylabel('Raw Traces')
%         plot(tracex,newtraces(:,1),'g');
%         hold on
%         plot(tracex,newtraces(:,3),'r');
%         hold on
%         yyaxis right
%         plot(tracex,(newtraces(:,4)),'b');
%         ylim([0,max(newtraces(:,4))*1.1])
%         ylabel('Ratio')
%         hold on
% 
%         title(strcat(musclename),tracename);
%         xlabel('Frames')    
% 
%         %time bar
%         xline(fr,'m','LineWidth',2);
% 
%         %save it temporarily for movie
%         f(fr) = getframe(fig); 
%         pause(0.1)
%         hold off
%     end
% 
%     hf = figure; 
% 
%     axis off
%     movie(hf,f);
%     mvi = mplay(f);
% 
%     v = VideoWriter(strcat(musclename,'.avi'));
%     v.FrameRate = 5;
%     open(v);
%     writeVideo(v,f);
% 
%     pause(1)
%     close(mvi)
%     close all
%     pause(2)
% end