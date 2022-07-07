%3D averaged movies code for manuscript

%uses Matlab videowriter to make movies from mat (ratio)
%allows for contrast manipulations, color scheme and bar, and inputs scale
% bars and time tickers

%PC, 05/2022
%Grueber Lab
%Columbia University

%% mat - ratio movie generation
clear all
close all

larvanames = {'larva2_run2','larva3_run1','larva4_run2_eL2','larva10_run1_L2','larva10_run4_L2'};
%image contrast params, customized across every ~50 frames per larva
clow = [80,80,100,80,100]; %1 = 80; 2 = 80; 3 = 100; 4 = 80; 5 = 100
chigh = [4400,4200,600,380,880];%1 = 4400; 2 = 4200; 3 = 600; 4 = 380; 5 = 880

%scale bar params for ums
pxumconv = 1.056;
ums = 100;
xsb = 1124;
ysb = 600;
col1 = round(xsb-(ums*pxumconv));
col2 = round(xsb+(ums*pxumconv));

%roll frames of the larvae for isolating those movies
rollframes = [[286,330];[6,45];[28,62];[400,420];[79,123]];

%caxis settings
clow = [80,80,100,80,100]; %1 = 80; 2 = 80; 3 = 100; 4 = 80; 5 = 100
chigh = [4400,4200,600,380,880];%1 = 4400; 2 = 4200; 3 = 600; 4 = 380; 5 = 880

for larva = 1:length(larvanames)
    %load data
    close all
    clear ratio
    clear M
    cd 'C:\Users\coone\Desktop\Patricia\newdataset-3D';
    larvafolder = [pwd,'\',larvanames{larva},'\final3davg-reapprois-0522'];
    cd(larvafolder);
    ratiof = load(strcat('fullrun_',larvanames{larva},'_ratio.mat'));
    ratio = ratiof.ratio;
    
    ratio(ratio < clow(larva)) = clow(larva);
    ratio(ratio > chigh(larva)) = chigh(larva);
    
    expername = larvanames{larva};
    roll = rollframes(larva,1):rollframes(larva,2);

    %go through all frames: plot + color
    for fr = 1:size(ratio,3)
        fig = figure;
        ratioframe = ratio(:,:,fr);
        imagesc(ratioframe);
        colormap('gray')
        axes = gca;
        axes.FontSize = 14;
        caxis([clow(larva),chigh(larva)]);
        colorbar('FontSize',14);
        hold on
        
        %add in scale bar and time ticker
        line([col1, col2], [ysb, ysb], 'Color','w','LineWidth',3);
        hold on
        
        disptime = fr/10;
        text(xsb-100,ysb-20,[num2str(disptime),' s'],'Color', 'w','FontSize',14);
        hold off
        
        %make full frame size and improve text res
        myfigsize = [1,41,1920,963];
        fig.Position = myfigsize;
        set(gcf,'PaperPosition',myfigsize);
        
        %capture each frame for movie
        M(fr) = getframe(gcf);
        close(fig)
        pause(2)
    end

    %write it to an .avi - 10fps and 5 fps
    vidname10 = strcat(expername,'_full_ratiomovie_10fps_gray.avi');
    vid10 = VideoWriter(vidname10,'Uncompressed AVI');
    set(vid10,'FrameRate',10);
    open(vid10);
    writeVideo(vid10,M)
    close(vid10);
    pause(30)
    
    vidname5 = strcat(expername,'_full_ratiomovie_5fps_gray.avi');
    vid5 = VideoWriter(vidname5,'Uncompressed AVI');
    set(vid5,'FrameRate',5);
    open(vid5);
    writeVideo(vid5,M)
    close(vid5);
    pause(30)
    
    %write just the roll frames to an .avi - 10fps and 5 fps
    vidname10 = strcat(expername,'_roll_ratiomovie_10fps_gray.avi');
    vid10 = VideoWriter(vidname10,'Uncompressed AVI');
    set(vid10,'FrameRate',10);
    open(vid10);
    writeVideo(vid10,M(roll))
    close(vid10);
    pause(30)
    
    vidname5 = strcat(expername,'_roll_ratiomovie_5fps_gray.avi');
    vid5 = VideoWriter(vidname5,'Uncompressed AVI');
    set(vid5,'FrameRate',5);
    open(vid5);
    writeVideo(vid5,M(roll))
    close(vid5);
    pause(30)
end