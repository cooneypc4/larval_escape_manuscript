% length v fluor analysis

%1. load in the mean fluor and length traces for each larva and muscle
%2. plot with simple line plots, all seg's separate --> 
    %show that mid segments are more important for roll than A and P 
    %(check that this is true for all muscles) - subplot per muscle,
    %left - left, mid - right, right - length
%3. boxplot of slopes for mid vs A vs P muscles -- quantitatively
%demonstrate above point

%PC - 11/22
%% load in the data for single muscle group, single larva
clear all
close all

larvanames = {'larva2_run2','larva3_run1'};
slopes = nan(6,6,length(larvanames)); %to put in slopes for all muscs, all segs, all larvae
%folds = nan(6,6,length(larvanames)); %to put in slopes for all muscs, all segs, all larvae

for larva = 1:length(larvanames)
    cd 'C:\Users\coone\Desktop\Patricia\newdataset-3D';
    larvafolder = [pwd,'\',larvanames{larva}];
    larvaf = dir([larvafolder '/**/','*redo1122-plusLTs']);
    cd(strcat(larvafolder,'\',larvaf.name));
    
    matfilenames = dir([pwd,'/*TRACK','*.mat']);
    
    for m = 1:length(matfilenames)
        clear rois
        clear ratio
        clear lengthy
        
        rois = load(matfilenames(m).name,'rois').rois;
        ratio = load(matfilenames(m).name,'ratio').ratio;
        lengthy = load(matfilenames(m).name,'lengthy').lengthy;
        muscle = string(matfilenames(m).name(27:29));
        if max(ratio(1:2,:)) == 0
            ratio(1:2,:) = nan;
            lengthy(1:2,:) = nan;
        end
        
%         %% plot lines for mean each muscle + length each muscle
%         figure
%         subplot(221)
%         imagesc(ratio);
%         title(strcat('Mean Ratio - ',muscle))
%         xlabel('Frames');
%         subplot(223)
%         vv = mean(ratio(3,find(ratio(3,:)>0)),'omitnan');
%         for j = 1:size(ratio,1)
%             if isnan(ratio(j,1))
%                 %skip
%             else
%                 tt = find(ratio(j,:)~=0);
%                 if find(diff(tt)>2)>0
%                     stoptt = find(diff(tt)>2);
%                     plot(tt(1:stoptt),ratio(j,1:stoptt)'-0.2*vv*j);
%                     hold on
%                     plot(tt(stoptt+1:end),ratio(j,tt(stoptt+1):end)'-0.2*vv*j);
%                     hold on
%                 else
%                     plot(tt,ratio(j,find(ratio(j,:)~=0))'-0.2*vv*j);
%                     hold on
%                 end
%             end
%         end
%         xlabel('Frames');
%         title(strcat('Mean Ratio - ',muscle))
%         subplot(222)
%         imagesc(lengthy);
%         xlabel('Frames');
%         title(strcat('Muscle Length - ',muscle))
%         subplot(224)
%         vv = mean(lengthy(3,find(lengthy(3,:)>0)),'omitnan');
%         for j = 1:size(lengthy,1)
%             if isnan(ratio(j,1))
%                 %skip
%             else
%                 tt = find(lengthy(j,:)~=0);
%                 if find(diff(tt)>2)>0
%                     stoptt = find(diff(tt)>2);
%                     plot(tt(1:stoptt),lengthy(j,1:stoptt)'-0.2*vv*j);
%                     hold on
%                     plot(tt(stoptt+1:end),lengthy(j,tt(stoptt+1):end)'-0.2*vv*j);
%                     hold on
%                 else
%                     plot(tt,lengthy(j,find(lengthy(j,:)~=0))'-1*vv*j);
%                     hold on
%                 end
%             end
%         end
%         title(strcat('Muscle Length - ',muscle))
%         xlabel('Frames');
% 
%         %save fig
%         saveas(gcf,strcat(larvanames{larva},'_',muscle,'_lengthvfluor'),'jpeg')
%         saveas(gcf,strcat(larvanames{larva},'_',muscle,'_lengthvfluor'),'svg')
%         saveas(gcf,strcat(larvanames{larva},'_',muscle,'_lengthvfluor','.fig'))
%         pause(1)
% 
%         
%         %% Easier to read plots: subplot for each of 6 segs, plot mean ratio on left y axis and length on right y axis, scale all to be same size
%         %traces - mean and length
%         figure('units','normalized','outerposition',[0 0 1 1])
%         for j = 1:size(lengthy,1)
%             subplot(6,1,j)
%             if isnan(ratio(j,1))
%                 %skip
%             else
%                 yyaxis left
%                 ylabel('Ratio')
%                 tt = find(lengthy(j,:)~=0);
%                 if find(diff(tt)>2)>0
%                     stoptt = find(diff(tt)>2);
%                     plot(tt(1:stoptt),ratio(j,1:stoptt)');
%                     hold on
%                     plot(tt(stoptt+1:end),ratio(j,tt(stoptt+1):end)');
%                     hold on
%                 else
%                     plot(tt,ratio(j,find(ratio(j,:)~=0))'-1);
%                     hold on
%                 end
%   
%                 yyaxis right
%                 ylabel('Length')
%                 if find(diff(tt)>2)>0
%                     stoptt = find(diff(tt)>2);
%                     plot(tt(1:stoptt),lengthy(j,1:stoptt)');
%                     hold on
%                     plot(tt(stoptt+1:end),lengthy(j,tt(stoptt+1):end)');
%                     hold on
%                 else
%                     plot(tt,lengthy(j,find(lengthy(j,:)~=0))'-1);
%                     hold on
%                 end
%             end
%         end     
%         xlabel('Frames');
%         sgtitle(strcat('Muscle Ratio Signal vs. Length - ',muscle))
%         
%         %save fig
%         saveas(gcf,strcat(larvanames{larva},'_',muscle,'_lengthvfluor_tracesep6'),'jpeg')
%         saveas(gcf,strcat(larvanames{larva},'_',muscle,'_lengthvfluor_tracesep6'),'svg')
%         saveas(gcf,strcat(larvanames{larva},'_',muscle,'_lengthvfluor_tracesep6','.fig'))
%         pause(1)
%         
        %store vals within new struct for comparison across larvae
        %store slopes for all muscles, columns = segments, row = slope observation
        
        for seg = 1:size(ratio,1)
            if isnan(ratio(seg,1))
                %skip
            else
                musc = ratio(seg,:);
                nonzero = musc(musc>0);
                descsort = sort(musc,'descend');
                slope = (descsort(1) - descsort(end)/descsort(1));
                %fold = descsort(1)/descsort(end);

                slopes(m,seg,larva) = slope;
                %folds(m,seg,larva) = fold;
            end
        end 
        
        
    end
end

%% boxplot and stats for slopes of ratio means across segments, combined across larvae
%combine larvae slopes from above
combslopes = cat(1,slopes(:,:,1),slopes(:,:,2));
%combfolds = cat(1,folds(:,:,1),folds(:,:,2));
segnames = [repmat({'A6'},1,12), repmat({'A5'},1,12), repmat({'A4'},1,12), repmat({'A3'},1,12), repmat({'A2'},1,12), repmat({'A1'},1,12)];

%boxplot
figure
boxplot(combslopes,segnames)
xlabel('Segments')
ylabel('Mean Ratio Slopes')
title('Dynamic Range across Segments during Rolling')
saveas(gcf,'lengthvfluor_slopes_z','jpeg')
saveas(gcf,'lenghtvfluor_slopes_z','svg')
saveas(gcf,'lengthvfluor_slopes_z.fig')

% figure
% boxplot(combfolds,segnames)
% xlabel('Segments')
% ylabel('Mean Ratio Fold Change')
% title('Dynamic Range across Segments during Rolling')
% saveas(gcf,'lengthvfluor_folds','jpeg')
% saveas(gcf,'lenghtvfluor_folds','svg')
% saveas(gcf,'lengthvfluor_folds.fig')

%kruskall wallis
[pacrosssegs,~,sgrp] = kruskalwallis(combslopes);
mc = multcompare(sgrp);

% [pfoldacrosssegs,~,fgrp] = kruskalwallis(combfolds);
% mc = multcompare(fgrp);