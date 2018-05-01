clear;

% Parameters
arrayName = 'DR';
staCodes = {'01','02','03','04','05','06','07','08','09','10','12'};
chans = 'SHE';
network = 'XG';
TDUR = 10.0;
filt = [2 8];

YY = 2010;
MM = 8;
DD = 16;
HH = 12;
mm = 30;
ss = 0;

Tstart = timeadd([YY MM DD HH mm ss]', - TDUR)';
Tend = timeadd([YY MM DD HH mm ss]', 60.0 + TDUR)';
opt = struct('time_tol', .01, 'fill_max', 3000000);

for ksta = 1 : length(staCodes)
    staList{ksta} = sprintf('%s%s.%s.%s', arrayName, staCodes{ksta}, chans, network);
end

% Preprocessing
D1 = getIRISdata(staList, Tstart, Tend, opt);
D2 = coralDetrend(D1);
D3 = coralTaper(D2, - 5);
for ksta = 1 : length(staCodes)
    D4(ksta) = coralDeconInst(D3(ksta), struct('opt', 'Deco from i to v cos .2 .5 10 15 wl 1e-8'));
end
D5 = coralFilter(D4, filt, 'bandpass');
D6 = coralResample(D5, struct('sintr_target', .05));

% Save data into mat files
for ksta = 1 : length(staCodes)
    filename1 = sprintf('matlab/%s%s_1.mat', arrayName, staCodes{ksta});
    time = - TDUR + D1(ksta).recSampInt * (0 : (size(D1(ksta).data, 1) - 1))';
    data = D1(ksta).data;
    save(filename1, 'time', 'data');
    filename2 = sprintf('matlab/%s%s_2.mat', arrayName, staCodes{ksta});
    time = - TDUR + D2(ksta).recSampInt * (0 : (size(D2(ksta).data, 1) - 1))';
    data = D2(ksta).data;
    save(filename2, 'time', 'data');
    filename3 = sprintf('matlab/%s%s_3.mat', arrayName, staCodes{ksta});
    time = - TDUR + D3(ksta).recSampInt * (0 : (size(D3(ksta).data, 1) - 1))';
    data = D3(ksta).data;
    save(filename3, 'time', 'data');
    filename4 = sprintf('matlab/%s%s_4.mat', arrayName, staCodes{ksta});
    time = - TDUR + D4(ksta).recSampInt * (0 : (size(D4(ksta).data, 1) - 1))';
    data = D4(ksta).data;
    save(filename4, 'time', 'data');
    filename5 = sprintf('matlab/%s%s_5.mat', arrayName, staCodes{ksta});
    time = - TDUR + D5(ksta).recSampInt * (0 : (size(D5(ksta).data, 1) - 1))';
    data = D5(ksta).data;
    save(filename5, 'time', 'data');
    filename6 = sprintf('matlab/%s%s_6.mat', arrayName, staCodes{ksta});
    time = - TDUR + D6(ksta).recSampInt * (0 : (size(D6(ksta).data, 1) - 1))';
    data = D6(ksta).data;
    save(filename6, 'time', 'data');
end
