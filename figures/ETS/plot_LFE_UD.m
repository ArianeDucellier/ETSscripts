% Plot figure of seismic signal to illustrate slide on LFEs

clear;

arrayName = 'DR';                   % name of array
staCodes = {'01','02','03','04','05','06','07','08','09','10','12'};
kchan = 'SHZ';
kname = 'Vertical';
network = 'XG';

TDUR = 10.0;                        % duration of data downloaded before and after window of interest
filt = [2 8];                       % band pass filter (Hz)
offset = 1e-6;                      % vertical offset for each seismogram
amp = 3.0;                          % amplification factor for seismogram

YY = 2010;
MM = 8;
DD = 16;
HH = 12;
mm = 30;
Tstart = timeadd([YY MM DD HH mm 0]', -TDUR)'; 
Tend = timeadd([YY MM DD HH mm 0]', TDUR + 60)'; 
opt = struct('time_tol', .01, 'fill_max', 3000000);
for ksta = 1 : length(staCodes)
    staList{ksta} = sprintf('%s%s.%s.%s', arrayName, staCodes{ksta}, kchan, network);
end
Dtmp = getIRISdata(staList, Tstart, Tend, opt); % get data from IRIS and put into coral structure
ntmp = [Dtmp.recNumData];
ktmp = find(ntmp == mode(ntmp));
Draw = Dtmp(ktmp); % remove stations that have different amounts of data
D = coralDetrend(Draw); % detrend data
D = coralTaper(D, -5); % taper first and last 5 s of data
D = coralResample(D, struct('sintr_target', .05)); % resample data to .05 s
% Remove instrument response
for k = 1 : length(D)
    D(k) = coralDeconInst(D(k), struct('opt', 'Deco from i to v cos .2 .5 10 15 wl 1e-8'));
end  
D = coralFilter(D, filt, 'bandpass'); % bandpass filter the data
optC = struct('cutType', 'absTime', 'absStartTime', [YY MM DD HH mm 0]', ...
                         'absEndTime', [YY, MM, DD, HH, mm + 1, 0]');
D = coralCut(D, optC);
t = (0 : length(D(1).data) - 1)' * D(1).recSampInt; % time vector for stacked data

clf;
figure(1);
subplot('Position', [0.1, 0.15, 0.8, 0.75]);
for ksta = 1 : length(D)
    plot(t, offset * ksta + amp * D(ksta).data, 'k');
    hold on;
end
xlim([0, 60]);
ylim([0, offset * (length(D) + 1)]);
xlabel('Time (s)');
ylabel('Velocity');
set(gca,'YTickLabel',[]);
set(gca,'FontSize', 20);
title(sprintf('Array %s - %s component', arrayName, kname));
eval(sprintf('print -depsc LFE_UD.eps'));
