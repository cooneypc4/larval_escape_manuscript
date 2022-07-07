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
clow = [80,20; 80,20; 40,20; 100,40; 200,80]; %row = larva, col 1 = gcamp, col 2 = mcherry
chigh = [4000,5200; 4000,8000; 500,64000; 60000,800; 70000,3600];

for larva = 1:length(larvanames)
    %load data
    close all
    clear M
    cd 'C:\Users\coone\Desktop\Patricia\newdataset-3D';
    larvafolder = [pwd,'\',larvanames{larva},'\final3davg-reapprois-0522'];
    cd(larvafolder);
    
    channelnames = {'gcamp', 'mcherry'};
    for channel = 1:2
        if channel == 1
            fname = strcat('fullrun_resize_rotate_pxthresh0.5_smooth5_win15_imageunsmooth_top_R_based_depth_gcamp_',larvanames{larva},'.tif');
        elseif channel == 2
            fname = strcat('fullrun_resize_rotate_pxthresh0.5_smooth5_win15_imageunsmooth_top_R_based_depth_mcherry_',larvanames{larva},'.tif');
        end
        tifdata = tiffLoad(fname);

        expername = larvanames{larva};
        roll = rollframes(larva,1):rollframes(larva,2);
        
        if channel == 1
            tifdata(tifdata < clow(larva,1)) = clow(larva,1);
            tifdata(tifdata > chigh(larva,1)) = chigh(larva,1);
        elseif channel == 2
            tifdata(tifdata < clow(larva,2)) = clow(larva,2);
            tifdata(tifdata > chigh(larva,2)) = chigh(larva,2);
        end

        %go through all frames: plot + color
        for fr = 1:size(tifdata,3)
            fig = figure;
            frame = tifdata(:,:,fr);
            imagesc(frame);
            colormap('gray')
            if channel == 1
                caxis([clow(larva,1),chigh(larva,1)]);
            else
                caxis([clow(larva,2),chigh(larva,2)]);
            end
            axes = gca;
            axes.FontSize = 14;
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
        vidname10 = strcat(expername,'_full_',channelnames{channel},'_movie_10fps_gray.avi');
        vid10 = VideoWriter(vidname10,'Uncompressed AVI');
        set(vid10,'FrameRate',10);
        open(vid10);
        writeVideo(vid10,M)
        close(vid10);
        pause(30)

        vidname5 = strcat(expername,'_full_',channelnames{channel},'_movie_5fps_gray.avi');
        vid5 = VideoWriter(vidname5,'Uncompressed AVI');
        set(vid5,'FrameRate',5);
        open(vid5);
        writeVideo(vid5,M)
        close(vid5);
        pause(30)

        %write just the roll frames to an .avi - 10fps and 5 fps
        vidname10 = strcat(expername,'_roll_',channelnames{channel},'_movie_10fps_gray.avi');
        vid10 = VideoWriter(vidname10,'Uncompressed AVI');
        set(vid10,'FrameRate',10);
        open(vid10);
        writeVideo(vid10,M(roll))
        close(vid10);
        pause(30)

        vidname5 = strcat(expername,'_roll_',channelnames{channel},'_movie_5fps_gray.avi');
        vid5 = VideoWriter(vidname5,'Uncompressed AVI');
        set(vid5,'FrameRate',5);
        open(vid5);
        writeVideo(vid5,M(roll))
        close(vid5);
        pause(30)
    end
end