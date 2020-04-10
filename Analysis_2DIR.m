%%	2D IR Analysis banana
%
%   This script Process, Analyze and Plot 2D data in time domain. 
%   It takes the EXPERIMENT FILE that contains the list of the data
%   files and the CALIBRATION FILE. It reads data and xml files that
%   contains the experiment details and saved processed data into four
%   matlab files. The outputs contains processed data, CLS and NLS
%   analysis results. This script also generates 2D IR plots and
%   line shape analysis (CLS&NLS) fittings.
%
%	Oct. 29nd, 2019     X.You
%
%   Jan. 17th, 2020  X.You;
%   Added negative delay referencing, SVD clean up function,
%   modified NLS and CLS analysis
%
%   April 3rd, 2020     X.You
%   Cleaned up for broader users, minor changes in interpolation

clear all;
close all;
clc;
%	For use on lab computer
% addpath('C:\Users\baizgroup\Box Sync\BaizGroup_Shared\Matlab');
addpath ('.\Functions')

%%	User Inputs - File Paths
% The directory you use to acess box
path.baseDir = 'C:\Users\cbaiz\Box\';
% Calibration file path in BOX
path.arrayfile = [path.baseDir 'BaizGroup_Shared\Data\2DIR\_calibration\mctArray\w3_1552_1755_23-Jan-2020.dat'];  % EtAc 2020.1.23 
%   Select the experiment file
expfile = 'C:\Users\cbaiz\Box\BaizGroup_Shared\Matlab\2DIR_Analysis_202004\Testing data\EtAc_D2O.txt';

%% Load the Calibration file
if ~isfile(path.arrayfile)
    [path.arrayfilename,path.arraypath] = uigetfile( 'C:\Users\cbaiz\Box\BaizGroup_Shared\Data\2DIR\sherry\*.dat', 'Please choose a calibration file');
    path.arrayfile = strcat(path.arraypath, path.arrayfilename);
end
w3=load(path.arrayfile);

%% Import Experiment file that contains records of data Files
%   For file loading
if isfile(expfile)
    [path.listpath,path.listname,path.listext] = fileparts(expfile);
    path.listfilename = strcat(path.listname,path.listext);
else
    [path.listfilename,path.listpath] = uigetfile( 'C:\Users\cbaiz\Box\BaizGroup_Shared\Data\2DIR\sherry\Experiment files\*.txt', 'Please choose an experiment file');
    expfile = strcat(path.listpath, path.listfilename);
    [path.listpath,path.listname,path.listext] = fileparts(expfile);
    
end
% Import file list
fileID=fopen(expfile);
filelist=textscan(fileID,'%s','Delimiter',',');
fclose(fileID);

%   For use on my laptop: If the experiment file folder is not in Box
%   Sync, then replace the file path with my local drive folder
if ~strcmp(path.baseDir, 'C:\Users\baizgroup\Box Sync\')
    filelist{1} = strrep(filelist{1},'C:\Users\baizgroup\Box Sync\',path.baseDir);
end

%	Saving options:creating a new folder
path.savepath = [path.listpath '\Processed Data\' path.listname];
mkdir(path.savepath)

clear fileID
%% Load .xml and .dat files
T2val = zeros (1,length(filelist{1}));
tic
for n = 1:length(filelist{1})
    [path.datapath{n},path.name{n},path.ext{n}] = fileparts(filelist{1}{n});
    path.xmlfile{n} = strcat(path.datapath{n},'\',path.name{n},'.xml');
    path.datafile{n}= strcat(path.datapath{n},'\',path.name{n},'.dat');
    
    xmlData{n}=	xmltools(path.xmlfile{n});
    
    %	Read T2
    T2xml{n}  =	xmlData{n}.child(2).child(8).child(2).value;
    T2val(n)  = str2num(T2xml{n});
    disp([num2str(n) ': T_2 = ' num2str(str2num(T2xml{n}),'%d')]);
    
    %   Read Description and obatain sample name
    Description = xmlData{1,1}.child(1).child(3).child(4).child(2).value;
    Description = strrep(Description,'w/',newline);
    Description = strrep(Description,'no',newline);
    Description = strrep(Description,'with',newline);
    Description = strsplit(Description,'\n');
    SampleNamexml{n} = Description{1};
    %   If the first line of the description is not the sample name, put
    %   in the sample name in the next line; extra space in the
    %   description will through error message to the save function (line 272)
     samplename = SampleNamexml{n};
    
    toc

    %   Load Data
    allData(:,:,n)   =	load( path.datafile{n} );
end

%%  User Inputs - Settings for data analysis and plottings

%	Using reference: ref = 1; Not using: ref = 0
ref = 0;

%	For plotting diagonal slice: diagOffset is how far off the
%	diagonal you want your slice to be (zero for right on the line)
diagOffset = 0;

%	Data collection info:
wRot =	1450;
dt =	20 ;
% finaldelay = 3000;

%	For plotting interferogram:
pixel = 72;

%   Super Gaussian parameters: test in 'single file processing' before
%   using the parameter
superGaussW = 2;  %%%%%%%%
superGaussP = 10;    %%%%%%%%

% Display range:
PumpCenter = 1690;
PumpWidth = 50;
ProbeCenter = 1690;
ProbeWidth = 50;

% Spectral range for 2D IR plots
AxSlim(1,:) = [PumpCenter - PumpWidth PumpCenter + PumpWidth];
AxSlim(2,:) = [ProbeCenter - ProbeWidth ProbeCenter + ProbeWidth];

% SVD cleanup range
AxTrim = [1600 2000];

% CLS and NLS analysis parameters
CLSbond = [1640 1750];
NLSbond = [1640 1750];
Cutoff = [50 4000];     % unit fs, cuttoff value for preliminary fitting
Threshold = 0.5;        % threshold for CLS peak filtering
InterpIndex = 0;        % Interpolation reduces the error bars in the fitting but increases processing time
BSNum = 3;              % Bootstrapping repetition

%   Record all settings to a file:
%save([ path.savepath '\' samplename '_user_input.mat'],'ref','w3','diagOffset','wRot', 'dt','superGaussW','superGaussP','samplename','titleref','PumpCenter','ProbeCenter','PumpWidth','ProbeWidth','AxSlim','AxTrim' );

%% %% %% Data processing %% %% %%
%% Create t2 delay series based on xml files
listDelays = sort(unique(T2val));

%%	Plots Titles/Info & Saving variables:
samplename = strtrim(SampleNamexml{1});
for n = 1:length(listDelays)
    savetitle{n} = [samplename '_T2_' num2str(abs(listDelays(n)))];%' num2str(T2val) 're' ];
end
if ref
    titleref=['_Referenced'];
else
    titleref=[''];
end

%% Select data range and do chop/phase cycle subtraction
wmin = ceil(min(w3));
wmax = ceil(max(w3));

% First 128 lines are raw data
rawData = allData(1:128,:,:);
rawData = circshift(rawData,[0 0]);

%	Next 128 lines are reference data
refData = zeros (size(rawData) );
refData = allData(129:256,:,:);

% refData = circshift(refData,[0 0]);
if ref
    [refMult, f] = Bvalues(allData,T2val);
    saveas( f,[path.savepath '\' samplename '_Bmatrix.png']);
else
    disp('Currently not using reference, see line 61')
    refMult = 0;
end

% Chop/phase cycle subtraction
shutterClosed = rawData(:,2:2:end,:);
shutterOpen   = rawData(:,1:2:end-1,:);
DeltaSignal = (log10(shutterOpen) - log10(shutterClosed));

%For reference shot-to-shot
shutterClosed1 = refData(:,2:2:end,:);
shutterOpen1   = refData(:,1:2:end-1,:);
DeltaRef = log10(shutterOpen1) - log10(shutterClosed1);

% Subtract reference data
DeltaSignalT=permute(DeltaSignal, [2 1 3]);
DeltaRefT=permute(DeltaRef, [2 1 3]);

for n = 1:size(allData,3)
    referenced(:,:,n) = DeltaSignalT(:,:,n) - DeltaRefT(:,:,n)*refMult;
end

% DeltaSignal'-DeltaRef'*ref;
dataref = (permute(referenced, [2 1 3]));
finaldelay = dt*((length(dataref(1,:)))-1);

data=dataref;

% Interpolate dead pixels
for n = 1:size(data,2)
    data(18,n,:)=mean([data(17,n,:) data(19,n,:)]);
    data(26,n,:)=mean([data(25,n,:) data(27,n,:)]);
    data(58,n,:)=mean([data(57,n,:) data(59,n,:)]);
    data(63,n,:)=mean([data(62,n,:) data(64,n,:)]);
    data(105,n,:)=mean([data(104,n,:) data(106,n,:)]);
end

% Subtract mean to zero the baseline
for n = 1:size(data,3)
    for j=1:size(data,2)
        data(:,j,n)=data(:,j,n)-mean(data(:,j,n));
    end
end

%% Zeropadding and time axis generation
data = data - repmat(mean(data,2), [1, size(data,2),1]);
zeroPadLen     = 1000;
zeroPadding    = repmat(data(:,end,:), [1,zeroPadLen]);
zeroPaddedData = permute([data zeroPadding],[2 1 3]);
numSteps       = size(zeroPaddedData,1);

t = ((0:dt:((numSteps*dt)-dt))/1000);
dnu = 1/(max(t)-min(t));
numax = (1/abs(t(2)-t(1)))/2;
fTHz = [0:dnu:numax -numax:dnu:-dnu];
w1Raw = fftshift(fTHz*33.356);

superGauss     = exp(-((t/superGaussW).^superGaussP));
superGaussMask = repmat(superGauss, [128 1])';

paddedMaskedData   = zeroPaddedData.*superGaussMask;
spectrum0          = real(fftshift(fft(paddedMaskedData),1));

%% Generate the w1 (pump) frequency axis
w3Rot = w3 - wRot;

[~, red] = min(abs(w3Rot(1) - w1Raw));
[~,blue] = min(abs(w3Rot(128) - w1Raw));

w1 = w1Raw(red:blue) + wRot;

%% Generate normalized 2D spectrum - no probe correction
spectrum1   = spectrum0(red:blue,:,:);
spectrum2   = (permute(spectrum1, [2 1 3])); %./(max(max(abs(spectrum1))));
freqAx = (wmin+1:wmax-1);

for n=1:size(data,3)
    IntSpec(:,:,n) = interp2(w1,w3',spectrum2(:,:,n),freqAx,freqAx');
    InterpProbeSpec(:,:,n) = interp1(w3, mean(shutterClosed(:,:,n),2), freqAx);
    normSpec(:,:,n) = repmat(InterpProbeSpec(:,:,n),[size(IntSpec,1) 1]);
    final2DSpec(:,:,n) = IntSpec(:,:,n);
end

%%  Call Data files at each T2 delay
freqAx = (wmin+1:wmax-1);
all2D_Data = final2DSpec(:,:,1:end);

sorted2D_Data = zeros (size(all2D_Data,1),size(all2D_Data,2),length(listDelays));
for n=1:length(listDelays)
    sorted2D_Data(:,:,n) = mean(all2D_Data(:,:,T2val(:) == listDelays(n)),3);
end

%%  Subtract scatter at negative delays
Scatter_Data = mean(all2D_Data(:,:,T2val<-500),3);
% sorted2D_Data = sorted2D_Data - Scatter_Data;

%    plot the scatter at negative delays
figure(2);clf;
contourf(freqAx,freqAx,Scatter_Data,[-1:0.1:1],'LineWidth',0.8);
colormap(cmap2d(50));
axis square;
line([wmin wmax], [wmin wmax], 'Color', [0 0 0], 'LineWidth', 1.5);
title([ samplename ' parallel' newline 'T2 < 0 ' ]);
xlabel('\omega_{1} (cm^{-1})');
ylabel('\omega_{3} (cm^{-1})');
saveas(gcf,[path.savepath '\' samplename '_2Dspec_negative_dalay' titleref '.png' ]);

%%  SVD Analysis
[U,S,V, svdAx, g] = SVD_Analysis(sorted2D_Data,freqAx,AxTrim);
saveas(g,[path.savepath '\' samplename ' SVD_components.png']);

%%  Apply SVD cleanup
[sorted2D_Data] = SVD_Cleanup(U,S,V,2,8,svdAx);
freqAx = svdAx;

%%	Define diagonal slice
%	Selects a slice parallel to the diagonal (ideally, right through
%	the max of the positive peak) - uses spectrum WITHOUT probe
%	normalization
normT2Data = zeros (size(sorted2D_Data));
shiftSpec = zeros (size(sorted2D_Data));
diagSlice = zeros (size(sorted2D_Data));

for n=1:size(sorted2D_Data,3)
    normT2Data(:,:,n) = sorted2D_Data(:,:,n)./max(max(abs(sorted2D_Data(20:end-20,20:end-20,n))));
    shiftSpec(:,:,n) = interp2(freqAx,freqAx',normT2Data(:,:,n),freqAx,(freqAx+diagOffset)');
    diagSlice(:,n) = diag(shiftSpec(:,:,n));   
end

%% Save Processed Data to the savepath as a '.mat' file
save([ path.savepath '\' samplename '.mat'], 'path','savetitle','data', 'dt', 'sorted2D_Data','normT2Data', 'shiftSpec','diagSlice','final2DSpec', 'freqAx', 'pixel', 'samplename', 'T2val', 'listDelays','titleref', 'wmin', 'wmax'); %[ savetitle '_CLS'],

%% %% %% PLOTS %% %% %%

%% Plot 2D spectrum - no probe correction
set(0,'DefaultFigureRenderer','painters');

for n = 1:size(normT2Data,3)
    k = n+2;
    figure(k); clf; box on;
    contourf(freqAx,freqAx,normT2Data(:,:,n),[-1:0.1:1],'LineWidth',0.7);
    colormap(cmap2d(50));
    axis square;
    line([wmin wmax], [wmin wmax], 'Color', [0 0 0], 'LineWidth', 1.5);
    title([ samplename newline 'T_2 = ' num2str(listDelays(n)) 'fs' ]);
    xlim([AxSlim(1,1) AxSlim(1,2)]);
    ylim([AxSlim(2,1) AxSlim(2,2)]);
    xlabel('\omega_{1} (cm^{-1})');
    ylabel('\omega_{3} (cm^{-1})');
    
    if listDelays(n)>0
        filename = [path.savepath '\' savetitle{n} '_2Dspec' titleref '.png' ];
%         filename = [path.savepath '\' savetitle{n} '_2Dspec' titleref '.png' ];
        saveas(gcf,filename);
    end
    close gcf
end

%% %% %% %% CLS and NLS Analysis %% %% %% %%

%% CLS Analysis and Preliminary Fitting
%   Trim data to a smaller window for faster fitting
Xindex = find((CLSbond(1) < freqAx) & (freqAx < CLSbond(2)));
Xax = freqAx(Xindex);
spec1 = normT2Data(Xindex,Xindex,:);

[ CLS_slope, stdev_CLS,Lower_CLS, Upper_CLS,PosDelays, h(1),h(2) ] = centerLineSlope( spec1, Xax, listDelays,Threshold, Cutoff);

%%	Bootstrapping/Jackknifing the fits
% Cutoff = [50 4000]; 
[lifetime_CLS,lifetimestd_CLS,offset_CLS,offsetstd_CLS,delta1_CLS ] = Bootstrap(CLS_slope,stdev_CLS,PosDelays,BSNum,Cutoff);

% Plot CLS Decay constant and offset
figure;
subplot(1,2,1); hold on; box on;
bar(1,lifetime_CLS,1,'DisplayName',[num2str(lifetime_CLS) ' ps']);
er = errorbar(1,[lifetime_CLS],[lifetimestd_CLS]');
er.Color = [0 0 0];
er.LineStyle = 'none';
title('CLS Decay Constant')
xlabel(samplename)
ylabel('Relaxation time (ps)')
%
subplot(1,2,2); hold on; box on;
bar(1,offset_CLS,1,'DisplayName',[num2str(offset_CLS) ' ps']);
er = errorbar(1,[offset_CLS],[offsetstd_CLS]);
er.Color = [0 0 0];
er.LineStyle = 'none';
title('CLS Offset')
xlabel(samplename)
ylabel('Offset (ps)')
hold off;
filename = [path.savepath '\' samplename '_Decay Constant_CLS.png'];
saveas(gcf,filename);

% Save CLS results
save([ path.savepath '\' samplename '_CLS.mat'],'spec1','PosDelays','Xax','Cutoff','Threshold','CLS_slope','stdev_CLS','Lower_CLS','Upper_CLS','lifetime_CLS','lifetimestd_CLS','offset_CLS','offsetstd_CLS');
saveas(h(1),[path.savepath '\' samplename '_CLS.png']);
saveas(h(2),[path.savepath '\' samplename '_CLS_Preliminary_Fitting.png']);
savefig(h,[ path.savepath '\' samplename '_CLS figures.fig'],'compact')

%% NLS Analysis and Preliminary Fitting
% Xindex = find((2000 < freqAx) & (freqAx < 2100));
Xindex = find((1640 < freqAx) & (freqAx < 1750));
Xax = freqAx(Xindex);
spec1 = normT2Data(Xindex,Xindex,:);

for n=1:size(spec1,3)
spec1(:,:,n) = spec1(:,:,n)./max(max(abs(spec1(:,:,n))));
end

[ NLSSlope,NLSstdev,LowerBound,UpperBound,PosDelays,l(1),l(2)] = NodalLineSlope(spec1, Xax, listDelays,InterpIndex,Cutoff);

%%	Bootstrapping/Jackknifing the fits
% Cutoff = [100 2000];
[NLSlifetime,NLSlifetimestd,NLSoffset,NLSoffsetstd,delta1_NLS ] = Bootstrap(NLSSlope,NLSstdev,PosDelays,BSNum,Cutoff);

% Plot NLS Decay constant and offset
figure;
subplot(1,2,1); hold on; box on;
bar(1,NLSlifetime,1,'DisplayName',[num2str(NLSlifetime) ' ps']);
er = errorbar([1],[NLSlifetime],[NLSlifetimestd]');
er.Color = [0 0 0];
er.LineStyle = 'none';
title('NLS Decay Constant')
xlabel(samplename)
ylabel('Relaxation time (ps)')

%
subplot(1,2,2); hold on; box on;
bar(1,NLSoffset,1,'DisplayName',[num2str(NLSoffset) ' ps']);
er = errorbar([1],[NLSoffset],[NLSoffsetstd]);
er.Color = [0 0 0];
er.LineStyle = 'none';
title('NLS Offset')
xlabel(samplename)
ylabel('Offset (ps)')
hold off;
filename = [path.savepath '\' samplename '_Decay Constant_NLS.png'];
saveas(gcf,filename);

% Save NLS results
save([ path.savepath '\' samplename '_NLS.mat'],'spec1','Xax','NLSSlope','NLSstdev','LowerBound','UpperBound','PosDelays','NLSlifetime','NLSlifetimestd','NLSoffset','NLSoffsetstd','NLSlifetime','delta1_NLS');
saveas(l(1),[path.savepath '\' samplename '_NLS.png']);
saveas(l(2),[path.savepath '\' samplename '_NLS_Preliminary_Fitting.png']);
savefig(l,[ path.savepath '\' samplename '_NLS figures.fig'],'compact')

%   SVD ANALYSIS FUNCTION
function [U,S,V, svdAx, f] = SVD_Analysis(Data,freqAx,Cutoff)
disp('Computing Singular Value Decomposition (SVD) ')

if ~exist('Cutoff','var')
    cutoffData = Data;
    svdAx = freqAx;
else
    cutoffData = Data(freqAx >= Cutoff(1) & freqAx <= Cutoff(2),freqAx >= Cutoff(1) & freqAx <= Cutoff(2),:);
    svdAx = freqAx(freqAx >= Cutoff(1) & freqAx <= Cutoff(2));
end
svdData = reshape(cutoffData,size(cutoffData,1)*size(cutoffData,2), size(cutoffData,3));
[U,S,V] = svd(svdData, 'econ');
f = figure('name', 'SVD components');clf;
for n = 1:5
    subplot(2,3,n);
    contourf(svdAx,svdAx,-reshape(U(:,n),size(cutoffData,1),size(cutoffData,2)),10,'LineWidth',0.8);
    colormap(cmap2d(50));
    axis square;
    xlabel('\omega_{1} (cm^{-1})');
    ylabel('\omega_{3} (cm^{-1})');
    title(['Component ',num2str(n)]);
end
subplot(2,3,6);
plot(diag(S),'-xr');
title('Singular values of each component');

end

%   SVD CLEANUP
function [CleanedData] = SVD_Cleanup(U,S,V,PCRemove,PCCut,svdAx)
if ~exist('PCRemove','var')
    PCRemove = 3;   % default to remove component 3 if no input
end
if ~exist('PCCut','var')
    PCCut = 6;   % default to neglect >6th components if no input
end
S(:,PCRemove)=0;
S(:,PCCut:end)=0;
svdCleaned = U*S*V';
CleanedData = reshape(svdCleaned,[length(svdAx), length(svdAx), size((svdCleaned),2)]);

% h = figure('name', 'SVD Cleanup');clf;
% for n = 1:size(CleanedData,3)
%     subplot(4,ceil(size(CleanedData,3)/4),n);
%     contourf(svdAx,svdAx,CleanedData(:,:,n),10,'LineWidth',0.8);
%     colormap(cmap2d(50));
%     axis square;
%     xlabel('\omega_{1} (cm^{-1})');
%     ylabel('\omega_{3} (cm^{-1})');
% end

end
%   BOOTSTRAPING FUNCTION
function [lifetime,lifetimestd,offset,offsetstd,delta1 ] = Bootstrap(Slope,stdev,Delays,BSNum,Cutoff)
disp('Bootstrapping...')

if ~exist('BSNum','var')
    BSNum = 2;
end

CutoffIndex = find(Delays > Cutoff(1)&Delays < Cutoff(2));

for n = 1:BSNum
    for m = 1:size(Slope,2)
        CLSTemp(m) = normrnd(Slope(m),stdev(m));
    end
    exponentialCurve = fit(Delays(CutoffIndex)',CLSTemp(CutoffIndex)',...
        'a*exp(-x/b)+c', 'StartPoint', [rand(), 500*rand(), rand()],'Lower',...
        [0, 100, 0],'Upper',[inf,inf,1]);
    coefficients = coeffvalues(exponentialCurve);
    BSOutput.lifetime(n) = coefficients(2);
    BSOutput.offset(n) = coefficients(3);
    BSOutput.delta1(n) = sqrt(coefficients(1));
end

lifetime = mean(BSOutput.lifetime)./1000;
lifetimestd = std(BSOutput.lifetime)./1000;
offset = mean(BSOutput.offset);
offsetstd = std(BSOutput.offset);
delta1 = mean(BSOutput.delta1);
end
