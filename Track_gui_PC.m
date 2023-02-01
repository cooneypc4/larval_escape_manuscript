function varargout = Track_gui_PC(varargin)
% TRACK_GUI_PC MATLAB code for Track_gui_PC.fig
%      TRACK_GUI_PC, by itself, creates a new TRACK_GUI_PC or raises the existing
%      singleton*.
%
%      H = TRACK_GUI_PC returns the handle to a new TRACK_GUI_PC or the handle to
%      the existing singleton*.
%
%      TRACK_GUI_PC('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRACK_GUI_PC.M with the given input arguments.
%
%      TRACK_GUI_PC('Property','Value',...) creates a new TRACK_GUI_PC or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Track_gui_PC_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Track_gui_PC_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Track_gui_PC

% Last Modified by GUIDE v2.5 02-Jun-2022 22:16:20
%edit for 2 sided ROI traces -- 11/22, PC

% Begin initialization code - DO NOT EDIT

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Track_gui_PC_OpeningFcn, ...
                   'gui_OutputFcn',  @Track_gui_PC_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Track_gui_PC is made visible.
function Track_gui_PC_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Track_gui_PC (see VARARGIN)

% Choose default command line output for Track_gui_PC
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Track_gui_PC wait for user response (see UIRESUME)
% uiwait(handles.main);


% --- Outputs from this function are returned to the command line.
function varargout = Track_gui_PC_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function Tslider_Callback(hObject, eventdata, handles)
% hObject    handle to Tslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
fr = round(handles.ss(end)*get(handles.Tslider,'Value'));
if fr ==0; fr = 1; end
set(handles.frame,'String',num2str(fr))
updateplot(hObject, handles)


% --- Executes during object creation, after setting all properties.
function Tslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Tslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function roinum_Callback(hObject, eventdata, handles)
% hObject    handle to roinum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of roinum as text
%        str2double(get(hObject,'String')) returns contents of roinum as a double


% --- Executes during object creation, after setting all properties.
function roinum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to roinum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in roiup.
function roiup_Callback(hObject, eventdata, handles)
% hObject    handle to roiup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
roi = str2num(get(handles.roinum,'String'));
set(handles.roinum,'String',num2str(roi+1));
updateplot(hObject,handles);
updateroi(hObject, handles)

% --- Executes on button press in roidown.
function roidown_Callback(hObject, eventdata, handles)
% hObject    handle to roidown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
roi = str2num(get(handles.roinum,'String'));
if roi>1;
set(handles.roinum,'String',num2str(roi-1));
end
updateplot(hObject,handles);
updateroi(hObject, handles)



% --- Executes on button press in select.
function select_Callback(hObject, eventdata, handles)
% hObject    handle to select (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.numpts,'Value'); numpts = 2; else numpts = 1; end
roi = str2num(get(handles.roinum,'String'));
frame = str2num(get(handles.frame,'String'));
[x y] = ginput_col(numpts);
handles.rois(roi,frame,1,:) = x;
handles.rois(roi,frame,2,:) = y;
handles.roitag(roi,frame) = 1;
guidata(hObject, handles)
if get(handles.interprois,'Value')
    if sum(handles.roitag(roi,:))>1;
    handles = interprois_Callback(hObject, eventdata, handles)
    end
end
updateroi(hObject, handles)
 guidata(hObject, handles)
 
        
function frame_Callback(hObject, eventdata, handles)
% hObject    handle to frame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of frame as text
%        str2double(get(hObject,'String')) returns contents of frame as a double
frnum =str2num(get(handles.frame,'String'));
if frnum==0; frnum = 1; end
if frnum>handles.ss(end); frnum = handles.ss(end); end
set(handles.Tslider,'Value',frnum/round(handles.ss(end)))
updateplot(hObject, handles)

% --- Executes during object creation, after setting all properties.
function frame_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file path] = uiputfile('*.mat', 'select save location');
roitag = handles.roitag_intp;
rois = handles.rois_intp;

%add length and signal measurements to saved file
for roi = 1:size(roitag,1)
    for frame = find(roitag(roi,:))
        x = round(squeeze(rois(roi,frame,1,:,:)));
        y = round(squeeze(rois(roi,frame,2,:,:)));
        mask = roipoly(squeeze(handles.data(:,:,frame)),[x(1) x(2) x(2) x(1)],[y(1)-1 y(2)-1 y(2)+1 y(1)+1]);
        ratio(roi,frame) = squeeze(sum(sum(double(squeeze(handles.data(:,:,frame))).*mask,2,'omitnan'),1,'omitnan'))./sum(sum(mask,'omitnan'),'omitnan');
        lengthy(roi,frame) =sqrt((x(1)-x(2)).^2+(y(1)-y(2)).^2);
    end
end

time_processed = datestr(now);
ss = handles.ss;
filename = get(handles.filepath,'String');

save([path 'TRACK_' file],'rois','time_processed','filename','roitag','ss','ratio','lengthy');

% --- Executes on button press in playmovie.
function playmovie_Callback(hObject, eventdata, handles)
% hObject    handle to playmovie (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.stoppy,'Visible','On','Value',0);
for i = 1:handles.ss(end)
    if get(handles.stoppy,'Value') == 0
    set(handles.frame,'String',num2str(i));
    frame_Callback(hObject, eventdata, handles)
    updateplot(hObject, handles)
    pause(0.1)
    end
end
set(handles.stoppy,'Visible','Off');

function ztop_Callback(hObject, eventdata, handles)
% hObject    handle to ztop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ztop as text
%        str2double(get(hObject,'String')) returns contents of ztop as a double


% --- Executes during object creation, after setting all properties.
function ztop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ztop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function zbot_Callback(hObject, eventdata, handles)
% hObject    handle to zbot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of zbot as text
%        str2double(get(hObject,'String')) returns contents of zbot as a double


% --- Executes during object creation, after setting all properties.
function zbot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zbot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in MIP.
function MIP_Callback(hObject, eventdata, handles)
% hObject    handle to MIP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of MIP


% --- Executes on button press in numpts.
function numpts_Callback(hObject, eventdata, handles)
% hObject    handle to numpts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of numpts



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filepath_Callback(hObject, eventdata, handles)
% hObject    handle to filepath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filepath as text
%        str2double(get(hObject,'String')) returns contents of filepath as a double


% --- Executes during object creation, after setting all properties.
function filepath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filepath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browse.
function browse_Callback(hObject, eventdata, handles)
% hObject    handle to browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile('*.mat', 'Pick a SCAPE mat file (2D or 3D)');
load([pathname, filename],'ratio');
handles.data = ratio;
set(handles.filepath,'String', [pathname, filename]);
handles.ss = size(handles.data);
clear ratio
guidata(hObject,handles)
updateplot(hObject, handles)

% --- Executes on button press in plottc.
function plottc_Callback(hObject, eventdata, handles)
% hObject    handle to plottc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.interprois,'Value',1);
handles = interprois_Callback(hObject, eventdata, handles)
win = [-2:2];
roitag = handles.roitag_intp;
rois = handles.rois_intp;
if size(rois,4)==1
    for roi = 1:size(roitag,1)
        for frame = find(roitag(roi,:))
            x = round(squeeze(rois(roi,frame,1,:)));
            y = round(squeeze(rois(roi,frame,2,:)));
            tc(roi,frame) = squeeze(mean(mean(handles.data(y+win, x+win,frame),2),1));
        end
    end
    figure
    subplot(121)
    imagesc(tc,[80,500]);
    xlabel('time');
    subplot(122)
    vv = mean(tc(1,find(tc(1,:)>0)));
    for j = 1:size(tc,1)
        tt = find(tc(j,:)~=0);
        plot(tt,tc(j,find(tc(j,:)~=0))'-0.2*vv*j);
        hold on
    end
    xlabel('time');
else
    for roi = 1:size(roitag,1)
        for frame = find(roitag(roi,:))
            x = round(squeeze(rois(roi,frame,1,:,:)));
            y = round(squeeze(rois(roi,frame,2,:,:)));
            mask = roipoly(squeeze(handles.data(:,:,frame)),[x(1) x(2) x(2) x(1)],[y(1)-1 y(2)-1 y(2)+1 y(1)+1]);
            tc(roi,frame) = squeeze(sum(sum(double(squeeze(handles.data(:,:,frame))).*mask,2,'omitnan'),1,'omitnan'))./sum(sum(mask,'omitnan'),'omitnan');
            lengthy(roi,frame) =sqrt((x(1)-x(2)).^2+(y(1)-y(2)).^2);
        end
    end
    figure
    subplot(221)
    imagesc(tc, [80,500]);
    title('mean signal')
    xlabel('time');
    subplot(223)
    vv = mean(tc(1,find(tc(1,:)>0)));
    for j = 1:size(tc,1)
        tt = find(tc(j,:)~=0);
        if find(diff(tt)>2)>0
            stoptt = find(diff(tt)>2);
            plot(tt(1:stoptt),tc(j,1:stoptt)'-0.2*vv*j);
            hold on
            plot(tt(stoptt+1:end),tc(j,tt(stoptt+1):end)'-0.2*vv*j);
            hold on
        else
            plot(tt,tc(j,find(tc(j,:)~=0))'-0.2*vv*j);
            hold on
        end
    end
    xlabel('time');
    title('mean signal')
    subplot(222)
    imagesc(lengthy);
    xlabel('time');
    title('length')
    subplot(224)
    vv = mean(lengthy(1,find(lengthy(1,:)>0)));
    for j = 1:size(lengthy,1)
        tt = find(lengthy(j,:)~=0);
        if find(diff(tt)>2)>0
            stoptt = find(diff(tt)>2);
            plot(tt(1:stoptt),lengthy(j,1:stoptt)'-0.2*vv*j);
            hold on
            plot(tt(stoptt+1:end),lengthy(j,tt(stoptt+1):end)'-0.2*vv*j);
            hold on
        else
            plot(tt,lengthy(j,find(lengthy(j,:)~=0))'-1*vv*j);
            hold on
        end
    end
    title('length')
    xlabel('time');
    %     subplot(325)
    %     plot(tc',lengthy');
    %     title('length')
    %     xlabel('time');
end


% --- Executes on button press in frameup.
        function frameup_Callback(hObject, eventdata, handles)
% hObject    handle to frameup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
frame = str2num(get(handles.frame,'String'));
if frame<handles.ss(end)
set(handles.frame,'String',num2str(frame+1));
set(handles.Tslider,'Value',frame/round(handles.ss(end)))
updateplot(hObject,handles);
updateroi(hObject, handles)
end


% --- Executes on button press in framedown.
function framedown_Callback(hObject, eventdata, handles)
% hObject    handle to framedown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
frame = str2num(get(handles.frame,'String'));
if frame>1
set(handles.frame,'String',num2str(frame-1));
set(handles.Tslider,'Value',frame/round(handles.ss(end)))
updateplot(hObject,handles);
updateroi(hObject, handles)
end


% --- Executes on button press in checkrois.
function checkrois_Callback(hObject, eventdata, handles)
% hObject    handle to checkrois (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure;
if isfield(handles,'roitag');
    imagesc(handles.roitag,[80,500])
    ylabel('rois');
    xlabel('frames');
end

% --- Executes on button press in loadrois.
function loadrois_Callback(hObject, eventdata, handles)
% hObject    handle to loadrois (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
out = 'Yes';
if isfield(handles,'roitag')
   out = questdlg('Warning, loading new ROIs will lose your current ROIs! Proceed?'); 
end
if strcmp(out,'Yes')
[file2 path] = uigetfile('*.mat', 'select tracking file');
load([path file2]);
handles.rois = rois;
handles.roitag = roitag;
% handles = rmfield(handles,'rois_intp');
% handles = rmfield(handles,'roitag_intp');
% time_processed = datestr(now);
% file = get(handles.filepath,'String');
% if strcmp(file,filename)~=1
%     errordlg(sprintf('warning, data filenames do not match (tracking from: %s)',filename));
% % set(handles.filepath,file);
% end
set(handles.interprois,'Value',0)
updateplot(hObject,handles);
updateroi(hObject, handles)
guidata(hObject,handles)
end

% --- Executes on button press in interprois.
function handles = interprois_Callback(hObject, eventdata, handles)
% hObject    handle to interprois (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.interprois,'Value')
    if size(handles.rois,4)==2; dims = 2; else; dims =1; end
    for roi = 1:size(handles.roitag,1)
        indy = find(handles.roitag(roi,:)==1);
        if length(indy)>1
            if find(diff(indy)>3)
                indystop = (find(diff(indy)>3,1,'last')); %add in ability to stop interpolating on off-screen frames, and restart later for same muscle
                stopind = indy(indystop);
                min1 = min(indy);%set here new min and max vals, such that goes from min to max pre-stop, then min to max post-stop diff 
                min2 = indy(indystop+1);
                for j = 1:2
                     for dimmy = 1:dims
                         roi_intp(roi,[min1:stopind],j,dimmy) = interp1(indy(1:indystop),handles.rois(roi,indy(1:indystop),j,dimmy),[min1:stopind]);
                         roi_intp(roi,[stopind+1:min2-1],j,dimmy) = nan;
                         roi_intp(roi,[min2:indy(end)],j,dimmy) = interp1(indy(indystop+1:end),handles.rois(roi,indy(indystop+1:end),j,dimmy),[min2:indy(end)]);
                     end
                end
                roitag_intp(roi,[min1:stopind]) = 1;
                roitag_intp(roi,[min2:indy(end)]) = 1;
            else
                for j = 1:2
                     for dimmy = 1:dims
                         roi_intp(roi,[min(indy):max(indy)],j,dimmy) = interp1(indy,handles.rois(roi,indy,j,dimmy),[min(indy):max(indy)]);
                     end
                end
                %fix whatever's happening for the muscles that only appear
                %once -- seems like roitag_intp is not saving right --
                %consider making the min(indy) different or max(indy)...?
                roitag_intp(roi,[min(indy):max(indy)]) = 1;
            end
        end
    end
    handles.rois_intp = roi_intp;
    handles.roitag_intp = roitag_intp;
    guidata(hObject,handles)
    updateroi(hObject, handles);
end
% Hint: get(hObject,'Value') returns toggle state of interprois
guidata(hObject,handles)

    function updateplot(hObject, handles)
        fr = str2num(get(handles.frame,'String'));
        axis(handles.xyplot);
       cla
       imagesc(squeeze(handles.data(:,:,fr)),[80,500]);
        set(gca,'Ycolor',[1 1 1],'Xcolor',[1 1 1]);
        axis on;
        colormap gray;
        updateroi(hObject, handles);
        
        
        function updateroi(hObject, handles);
            if isfield(handles,'rois');
                if get(handles.showall,'Value')
                    roiuse = 1:size(handles.roitag,1);
                else
                    roiuse = str2num(get(handles.roinum,'String'));
                end
                for roi = roiuse
                    frame = str2num(get(handles.frame,'String'));
                    roitag = handles.roitag;
                    rois = handles.rois;
                    if roi<=size(roitag,1) && frame<=size(roitag,2);
                        if get(handles.interprois,'Value') && sum(handles.roitag(roi,:))>1;
                            roitag = handles.roitag_intp;
                            rois = handles.rois_intp;
                            
                        end
                        if roitag(roi,frame)>0;
                            x = squeeze(rois(roi,frame,1,:));
                            y = squeeze(rois(roi,frame,2,:));
                            axes(handles.xyplot)
                            hold on
                            plot(x,y,'-or');%'color',[1 0 1],'MarkerSize',12)
                            text(x+4,y+4,num2str(roi),'color',[1 0 1]);
                            hold off
                        end
                    end
                end
            end

% --- Executes on button press in stoppy.
function stoppy_Callback(hObject, eventdata, handles)
% hObject    handle to stoppy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of stoppy


% --- Executes on button press in showall.
function showall_Callback(hObject, eventdata, handles)
% hObject    handle to showall (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of showall
if get(handles.showall,'Value');
    updateroi(hObject, handles);
end
