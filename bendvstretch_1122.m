%plot DR/R for diff moments of rolling for all muscles; categ vs. contin

%quantitative comparisons
%1. loads in data from segmented muscles of each larva
%2. finds centroid of each ROI over time
%3. plot bend side vs. curve side activity:
    %b. categorize: muscle peak <=50% muscle present, >50% muscle present
        %i. divide by bend-roll type:
            %Take values from % of non-zero trace timepoints
    %c. take mean of ratiometric val's for frames divided into 4 sections
    %of bend vs. stretch
    %d. plotboxplot of the data 
        %i. grouped by muscle, D-V for all larvae, w and w/o norm
        %ii. grouped by SMUG, D-V for all larvae, w and w/o norm
    %e. mult anova

        
% 4. align all muscle traces via interpolation
% 5. plot group lines and individual muscle lines with SEM bar and do stats

%Patricia Cooney, 04/2022, 5/2022, 6/2022, 11/2022
%Grueber Lab
%Columbia University
 
%% load ROIs per larva
clear all
close all

larvanames = {'larva2_run2','larva3_run1'};%,'larva10_run1_L2'};%'larva4_run2_eL2',,'larva10_run4_L2'};
color = {'r','b','m','g','k'};
dx = -0.001;
dy = -0.001;

DLs = {'01','03','09','10'};
DLinds = 1:4;
DOs = {'11','19','20'};
DOinds = 5:7;
LLs = {'04','05','12'};
LLinds = 8:10;
LTs = {'08','21','22','23'};
LTinds = 11:14;
VAs = {'26','27'};
VAinds = 15:16;
VOs = {'15','16','17'};
VOinds = 17:19;
allmuscs = [DLs, DOs, LLs, LTs, VAs, VOs]; %indices here are in order of DV
smuggrps = {'DLs', 'DOs', 'LLs', 'LTs', 'VAs', 'VOs'};
lengthmuscs = max([length(DLinds), length(DOinds), length(LLinds), length(LTinds), length(VAinds), length(VOinds)]);

ratiomois = nan(length(allmuscs),2,2,3,length(larvanames)); %muscle,medians,lr,seg,larva

for larva = 1:length(larvanames)
    clear meanratiobend
    clear meanratiostretch
    cd 'C:\Users\coone\Desktop\Patricia\newdataset-3D';
    larvafolder = [pwd,'\',larvanames{larva}];
    larvaf = dir([larvafolder '/**/','*redo1122-plusLTs']);
    cd(strcat(larvafolder,'\',larvaf.name));
    
    matfilenames = dir([pwd,'/*recalcreapp','*_avgdata.mat']);
    test = load(matfilenames(1).name);
    frames = length(test.avgdata.newtraces);
  
    for m = 1 : length(matfilenames)
        clear loaded
        loaded = load(matfilenames(m).name,'avgdata');
        newtraces = loaded.avgdata.newtraces; 
        ratio = (newtraces(:,4));

        if larva == 3 || larva == 5
            roll = 2;
        elseif larva == 1 || larva == 2 || larva == 4
            roll = 1;
        end

        musclename = matfilenames(m).name(13:17);

        if max(contains(allmuscs(:),musclename(4:5))>0)
            moi = 1;
            muscnum = find(contains(allmuscs(:),musclename(4:5)));
            if contains(musclename,'l')
                lr = 1;
            else
                lr = 2;
            end
            seg = str2double(musclename(2))-1;
            if seg == 4 && muscnum == 26
                seg = 3;
            elseif seg == 4
                seg = 1;
            elseif seg == 5
                seg = 2;
            end
        end
        
        %% take the 20% edges of non-zero trace for bend vs stretch values
        %separate into single bump trace vs. two bumps first
        nonzeroinds = find(ratio > 0);
        nonzero = ratio(ratio > 0);
        %nonzero = ratio(ratio>0);
        bendinds = [];
        stretchinds = [];
        if find(ratio<=0,1,'first') > find(ratio>0,1,'first')
            traceend1 = nonzeroinds(diff(nonzeroinds)>1);
            if length(traceend1)>1
                traceend1 = traceend1(1);
            end
            tracestart2 = nonzeroinds(traceend1 + 1);
            length2 = frames - tracestart2 + 1;
            
            if roll == 1
                bendinds = 1:round(traceend1*.1);
                stretchinds = traceend1-round(traceend1*.1):traceend1;
                
                if length2 > 4
                    bendinds = [bendinds, tracestart2 : tracestart2 + round(length2*0.1)];
                    stretchinds = [stretchinds frames - round(length2*-0.1) : frames];
                else
                    bendinds = [bendinds, tracestart2];
                end
            elseif roll == 2
                stretchinds = 1:round(traceend1*.1);
                bendinds = traceend1-round(traceend1*.1):traceend1;
                
                if length2 > 4
                    stretchinds = [stretchinds, tracestart2 : tracestart2 + round(length2*0.1)];
                    bendinds = [bendinds, frames - round(length2*-0.1) : frames];
                else
                    stretchinds = [stretchinds, tracestart2];
                end
            end
            ratio = [normalize(ratio(1:traceend1),1); zeros(tracestart2-1-traceend1+1,1); normalize(ratio(tracestart2:end),1)];
        else
            tracestart1 = nonzeroinds(1);
            traceend1 = nonzeroinds(end);
            length2 = traceend1 - tracestart1 + 1;
            if roll == 1
                bendinds = tracestart1:tracestart1+round(length2*.2);
                stretchinds = traceend1-round(length2*.2):traceend1;
            elseif roll == 2
                stretchinds = tracestart1:tracestart1+round(length2*.2);
                bendinds = traceend1-round(length2*.2):traceend1;
            end
            ratio = [zeros(tracestart1-1,1); normalize(ratio(tracestart1:traceend1),1); zeros(traceend1+1,1)];
        end

        meanratiobend = mean(ratio(bendinds),'omitnan');
        meanratiostretch = mean(ratio(stretchinds),'omitnan');

        ratiomois(muscnum,:,lr,seg,larva) = [meanratiobend, meanratiostretch]; %muscle,medians,lr,seg,larva
        
    end    
    %take min and max ratio vals per larva and scale so every larva's
    %muscles vary in ratio values from 0 to 1, cleaner comparison
%     maxlarv = max(ratiomois(:,:,:,:,larva));
%     minlarv = min(ratiomois(:,:,:,:,larva));
%     
    %normalize larva's data to minmax of larva
    %normratiomois(:,:,:,:,larva) = (ratiomois(:,:,:,:,larva) - minlarv)./(maxlarv - minlarv);
    
    for m = 1:size(ratiomois,1)
     %% figures
     %plot unity line + ratiocomp for indiv larva
        figure(larva)
        scatter(reshape(ratiomois(m,1,:,:,larva),[],1),reshape(ratiomois(m,2,:,:,larva),[],1),color{larva})
        text(reshape(ratiomois(m,1,:,:,larva),[],1)+dx,reshape(ratiomois(m,2,:,:,larva),[],1)+dy, matfilenames(m).name(13:17))
        hold on
        
        figure(8)
        scatter(reshape(ratiomois(m,1,:,:,larva),[],1),reshape(ratiomois(m,2,:,:,larva),[],1),color{larva})
        text(reshape(ratiomois(m,1,:,:,larva),[],1)+dx,reshape(ratiomois(m,2,:,:,larva),[],1)+dy, matfilenames(m).name(13:17))
        hold on
    end
    
    %finish out the figure details - old scatter unity plot
    figure(larva); %all for single larva
    %make unity line
    xvals = [-2,2];
    yvals = [-2,2];
    plot(xvals,yvals,'--')
    title(strcat('Ratio Values - ',string(larvanames{larva})))
    xlabel('Mean Ratio for Frames in Bend')
    ylabel('Mean Ratio for Frames in Stretch')
    hold off
    saveas(gcf,strcat(string(larvanames{larva}),'mean_ratio_unityplots_zscore_remla10-4'),'jpeg')
    saveas(gcf,strcat(string(larvanames{larva}),'mean_ratio_unityplots_zscore_remla10-4','.fig'))
    saveas(gcf,strcat(string(larvanames{larva}),'mean_ratio_unityplots_zscore_remla10-4'),'svg')
    
end

%finish out the figure details
figure(8) %all
plot(xvals,yvals,'--')
title('Mean Values for All Larvae')
xlabel('Mean Ratio for Frames in Bend')
ylabel('Mean Ratio for Frames in Stretch')
saveas(gcf,'all_mean_ratio_unityplots_zscore_remla10-4','jpeg')
saveas(gcf,'all_mean_ratio_unityplots_zscore_remla10-4.fig')
saveas(gcf,'all_mean_ratio_unityplots_zscore_remla10-4','svg')
%%

%plot averages of all larvae
%ratiomois(muscnum,:,lr,seg,larva) = [medianratiobend(a),medianratiostretch(a)]; %muscle,means,lr,seg,larva
nor_normmeanratiomois = mean(ratiomois,5,'omitnan');
nor_normmeancombratiomois = mean(nor_normmeanratiomois,4,'omitnan'); %remove grp by segs
nor_normmeansinglevalratiomois = mean(nor_normmeancombratiomois,3,'omitnan'); %remove LR

figure(9)
colorseg = {'b','g','m'};
side = {'l','r'};
for se = 1:3
    for si = 1:2
        for mn = 1:size(nor_normmeanratiomois,1)
            scatter(nor_normmeanratiomois(mn,1,si,se),nor_normmeanratiomois(mn,2,si,se),colorseg{se});
            text(nor_normmeanratiomois(mn,1,si,se)+dx, nor_normmeanratiomois(mn,2,si,se)+dy,strcat(num2str(se+1),side{si},allmuscs{mn}));
            hold on
        end
    end
end

figure(9) %avg all w/ seg and side
plot(xvals,yvals,'--')
title('Combined Mean Ratio Values for All Larvae')
xlabel('Average of Mean Ratio for Frames in Bend')
ylabel('Average of Mean Ratio for Frames in Stretch')
saveas(gcf,'mean_ratio_unityplots_avgallbyseg_redo_zscore_remla10-4','jpeg')
saveas(gcf,'mean_ratio_unityplots_avgallbyseg_redo_zscore_remla10-4.fig')
saveas(gcf,'mean_ratio_unityplots_avgallbyseg_redo_zscore_remla10-4','svg')

%average all (combined seg and side)
figure(10)
for mn = 1:size(nor_normmeansinglevalratiomois,1)
    scatter(nor_normmeansinglevalratiomois(mn,1),nor_normmeansinglevalratiomois(mn,2));
    text(nor_normmeansinglevalratiomois(mn,1)+dx, nor_normmeansinglevalratiomois(mn,2)+dy,allmuscs{mn});
    hold on
end

figure(10) %avg all
plot(xvals,yvals,'--')
title('Combined Mean Ratio Values for All Larvae')
xlabel('Average of Mean Ratio for Frames in Bend')
ylabel('Average of Mean Ratio for Frames in Stretch')
saveas(gcf,'mean_ratio_unityplots_avgallcombined_redo_zscore_remla10-4','jpeg')
saveas(gcf,'mean_ratio_unityplots_avgallcombined_redo_zscore_remla10-4.fig')
saveas(gcf,'mean_ratio_unityplots_avgallcombined_redo_zscore_remla10-4','svg')

%% post hoc regression to show diff b/t unity and actual activity
mdl = fitlm(nor_normmeansinglevalratiomois(:,1),nor_normmeansinglevalratiomois(:,2));
figure(9)
plot(mdl)

figure(10)
plot(mdl)
save('forregression_unityinfo_quarters_allcomb_zscore_remla10-4.mat','mdl');


%% now group, plot, and stats categorically w/ beehive and box
%group together
%ratiomois(muscnum,:,lr,seg,larva) = [meanratiobend, meanratioprebend, meanratioprestretch, meanratiostretch]; %muscle,medians,lr,seg,larva
%20, 4, 2, 3, 2-5

%make a new matrix for each muscle where col's are categ's
bymusc = nan(2 * 3 * length(larvanames),2,length(allmuscs));
for im = 1:length(allmuscs)
    it = 1;
    for lar = 1:length(larvanames)
        for seg = 1:3
            for lr = 1:2
                bymusc(it,:,im) = ratiomois(im,:,lr,seg,lar); %put muscles into col grpings
                it = it + 1;
            end
        end
    end
end
%grp and alize - range v zscore
bymusccols = reshape(bymusc,size(bymusc,1),2*length(allmuscs));

d = 1;
do = 1;
ll = 1;
lt = 1;
va = 1;
vo = 1;

bygrp = nan(size(bymusc,1)*5,length(smuggrps)*size(bymusc,2));

for mg = 1:length(allmuscs)
    if ismember(mg,DLinds)
        bygrp(d*size(bymusccols,1) - (size(bymusccols,1)-1):d*size(bymusccols,1),1:2) = bymusc(:,:,mg);
        d = d+1;
    elseif ismember(mg,DOinds)
        bygrp(do*size(bymusccols,1) - (size(bymusccols,1)-1):do*size(bymusccols,1),3:4) = bymusc(:,:,mg);
        do = do+1;
    elseif ismember(mg,LLinds)
        bygrp(ll*size(bymusccols,1) - (size(bymusccols,1)-1):ll*size(bymusccols,1),5:6) = bymusc(:,:,mg);
        ll = ll+1;
    elseif ismember(mg,LTinds)
        bygrp(lt*size(bymusccols,1) - (size(bymusccols,1)-1):lt*size(bymusccols,1),7:8) = bymusc(:,:,mg);
        lt = lt+1;
    elseif ismember(mg,VAinds)
        bygrp(va*size(bymusccols,1) - (size(bymusccols,1)-1):va*size(bymusccols,1),9:10) = bymusc(:,:,mg);
        va = va+1;
    elseif ismember(mg,VOinds)
        bygrp(vo*size(bymusccols,1) - (size(bymusccols,1)-1):vo*size(bymusccols,1),11:12) = bymusc(:,:,mg);
        vo = vo+1;
    end
end
%bygrp(bygrp==0)=nan;

%stats all - COME BACK TO THIS AND JUST COMPARE BEND V STRETCH W/IN MUSCLES AND THEN WITHIN GRPS, PULL P'S
[pwinmusc,~,sindmu] = kruskalwallis(bymusccols);
cind = multcompare(sindmu);
% pbymusc = nan(length(allmuscs));
% for pm = 1:length(allmuscs)
%     checkind = pm
%     pbymusc(pm) = cind(pm+4,6);
% end

[pacrossmusc,~,sgrp] = kruskalwallis(bygrp);
cgrp = multcompare(sgrp);

%reshape everything for plotting
indivmuscs = reshape(bymusccols,[],1);
grpmuscs = reshape(bygrp,[],1);

%for indiv groupings
migrps = cell(1,length(indivmuscs));
for mu = 1:length(allmuscs)
    migrps(mu*size(bymusccols,1)*2-(size(bymusccols,1)*2-1):mu*size(bymusccols,1)*2) = {char(allmuscs(mu))};
end
migrps = migrps';
cols = repmat([repmat({'bend'},1,size(bymusccols,1)),...
    repmat({'stretch'},1,size(bymusccols,1))],1,length(allmuscs))';

%for smug-like groupings
sgrps = cell(1,length(grpmuscs));
for sg = 1:length(smuggrps)
    sgrps(sg*size(bymusccols,1)*2*5-(size(bymusccols,1)*5*2-1):sg*size(bymusccols,1)*5*2) = {char(smuggrps(sg))};
end
sgrps = sgrps';
colgs = repmat([repmat({'bend'},1,size(bymusccols,1)*5),...
    repmat({'stretch'},1,size(bymusccols,1)*5)],1,length(smuggrps))';

%box and bee
%raw
figure
boxplot(indivmuscs,{migrps,cols},"ColorGroup",cols,'PlotStyle','compact')
xlabel('Muscles during Roll Phases')
ylabel('Mean Ratiometric Signal')
title(strcat('Mean Ratiometric Signal during Rolling'))
saveas(gcf,'allmuscs_meanratioquarts_all_zscore_remlar10-4','jpeg')
saveas(gcf,'allmuscs_meanratioquarts_all_zscore_remlar10-4.fig')

%by grp
%raw
figure
boxplot(grpmuscs,{sgrps,colgs},"ColorGroup",colgs,'PlotStyle','compact')
xlabel('Muscle Groups during Roll Phases')
ylabel('Mean Ratiometric Signal')
title(strcat('Mean Ratiometric Signal during Rolling'))
saveas(gcf,'muscgrps_meanratioquarts_all_zscore_remlar10-4','jpeg')
saveas(gcf,'muscgrps_meanratioquarts_all_zscore_remlar10-4.fig')
% 
% %check indiv effects
% [pwinmusc,~,sindmu] = kruskalwallis(indivmuscs,cols);
% cind = multcompare(sindmu);
% [pxmusc,~,sindmux] = kruskalwallis(indivmuscs,migrps);
% cgind = multcompare(sindmux);
% 
% [pacrossmusc,~,sgrp] = kruskalwallis(grpmuscs,colgs);
% cgrp = multcompare(sgrp);
% [pxgmusc,~,sxindmu] = kruskalwallis(grpmuscs,sgrps);
% csxind = multcompare(sxindmu);


% %% try division and subtraction metrics within larva for each averaged muscle
% 
% 
% coldivratios = reshape(divratiomois',[],1);
% colsubratios = reshape(subratiomois',[],1);
% colmuscs = cell(1,length(coldivratios));
% for cm = 1:length(allmuscs)
%     colmuscs(cm*5-4:cm*5) = {char(allmuscs(cm))};
% end
% 
% %% plotting divisions and subtractions
% figure
% boxplot(coldivratios,colmuscs);
% title('div all - norm_rescale')
% saveas(gcf,'all_div_raw_remla-4-10-4','jpeg')
% saveas(gcf,'alldiv_raw_remla-4-10-4.fig')
% [pdivmuscs,~,sdivmuscs] = kruskalwallis(divratiomois');
% divind = multcompare(sdivmuscs);
% 
% figure
% boxplot(colsubratios,colmuscs);
% title('sub all - norm_rescale')
% saveas(gcf,'all_sub_raw_remla-4-10-4','jpeg')
% saveas(gcf,'allsub_raw_remla-4-10-4.fig')
% [psubmuscs,~,ssubmuscs] = kruskalwallis(subratiomois');
% subind = multcompare(ssubmuscs);