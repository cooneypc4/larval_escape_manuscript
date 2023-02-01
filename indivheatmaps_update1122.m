%% Individual Heatmaps for each larva

%heatmaps for all ratio traces in most consistently measured muscles

%Patricia Cooney, 6/2022
%Grueber Lab
%Columbia University

%% load ROIs per larva
clear all
close all

larvanames = {'larva2_run2','larva3_run1','larva4_run2_eL2','larva10_run1_L2','larva10_run4_L2'};
larvae = struct;

%pull out the muscles of interest (most consistently measured)
DLs = {'01','10'};%'03','09'
DLinds = 1:2;
DOs = {'11','19'};%,'20'};
DOinds = 3:4;
LLs = {'04','05'};%,'12'};
LLinds = 5:6;
LTs = {'21','22','23'};%'08',
LTinds = 7:9;
VAs = {'26','27'};
VAinds = 10:11;
VOs = {'15','16'};%,'17'};
VOinds = 12:3;
allmuscs = [DLs, DOs, LLs, LTs, VAs, VOs]; %indices here are in order of DV
smuggrps = {'DLs', 'DOs', 'LLs', 'LTs', 'VAs', 'VOs'};
lengthmuscs = max([length(DLinds), length(DOinds), length(LLinds), length(LTinds), length(VAinds), length(VOinds)]);

%cax = [0, 2.4];

for larva = 1:length(larvanames)    
    clear frames
    cd 'C:\Users\coone\Desktop\Patricia\newdataset-3D';
    larvafolder = [pwd,'\',larvanames{larva}];
    larvaf = dir([larvafolder '/**/','*redo1122-plusLTs']);
    cd(strcat(larvafolder,'\',larvaf.name));

    matfilenames = dir([pwd,'/*recalcreapp','*_avgdata.mat']);
    test = load(matfilenames(20).name);
    frames = length(test.avgdata.newtraces);
    
    allratios = nan(length(allmuscs),frames,2,3); %musclenum,maxtracelength,lr,seg

    for m = 1:length(matfilenames)
        clear ratio
        clear musclename
        musclename = matfilenames(m).name(13:17);
        if max(contains(allmuscs(:),musclename(4:5))>0)
            loaded = load(matfilenames(m).name,'avgdata');
            newtraces = loaded.avgdata.newtraces; 
            ratio = (newtraces(:,4)'); %make it a row

            %sort out details for indexing
            seg = str2double(musclename(2))-1;
            muscnum = find(contains(allmuscs(:),musclename(4:5)));
            if contains(musclename,'l')
                lr = 1;
            else
                lr = 2;
            end
              
            allratios(muscnum,:,lr,seg) = ratio;
            normratios = normalize(allratios,2);
        end
    end
    
    %heatmap for indiv larvae: all muscs, sep segs, sep sides
    figure
    
    pp = parula;
    pp(1:42,:) = 0; %set frames where muscle is out of field to black
    
    
    for lr = 1:2
        for seg = 1:3
            if lr == 2
                subplot(3,2,lr*seg)
            else
                subplot(3,2,(lr+1)*seg-1)
            end
            imagesc(normratios(:,:,lr,seg))
            colormap(pp)
            yticks(1:length(allmuscs))
            yticklabels(allmuscs);
            %caxis(cax);
            colorbar('FontSize',14);
            if lr == 1 && seg == 1
                title('Left Side')
            elseif lr == 2 && seg == 1
                title('Right Side')
            elseif seg == 2
                ylabel('Muscle Ratio Traces')
            elseif seg == 3
                xlabel('Frames')
            end
            hold on
        end
        hold on 
    end
    sgtitle(strcat('Heat Map of Muscles across Segments - ',larvanames{larva}));
    
    %SAVE
    saveas(gcf,strcat(larvanames{larva},'_indivheatmaps_black_zscore_bthresh_subset'),'jpeg')
    saveas(gcf,strcat(larvanames{larva},'_indivheatmaps_black_zscore_bthresh_subset'),'svg')
    saveas(gcf,strcat(larvanames{larva},'_indivheatmaps_black_zscore_bthresh_subset','.fig'))
end
save('alllarvae_allratios_zscore_bthresh.mat','larvae');