clc;
clear all;
%This script will vary the  and constructs the received signal rx. Then,
%stores the modulation data for each rx in a specific data structure.
%% Initialization of moduation types
modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
    "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK"]);

%% Center frequency of the transmit and receive antenna arrays
fc=70e9;% Center frequencies
lambda=3e8/fc; %Wavelength

%% Initialization of spacing btween two adjacent antennas
antennaSpacing = linspace(0.1*lambda,lambda,10);
for m=1:length(antennaSpacing)

    %Here the electromagnetic will be introduced like this for various antennas
    %and materials and different array topology

    ant=design(linearArray,fc);
    ant.ElementSpacing=antennaSpacing(m);
    ant.NumElements=128;
    zin=impedance(ant,fc);
    S=10*log10(sum(real(10./conj(zin))));
    N=-10;%in dB
    SNR(m)=S-N;
end

%% Initialize channel parameters
sps = 8;                % Samples per symbol
spf = 1024;             % Samples per frame
symbolsPerFrame = spf / sps;
if fc==fc
fs = 800e6;             % Sample rate in
else
fs = 100e6;             % Sample rate in
end
maxDeltaOff = 5;
deltaOff = (rand()*2*maxDeltaOff) - maxDeltaOff;
C = 1 + (deltaOff/1e6);

%% Set the random number generator to a known state to be able to regenerate
% the same frames every time the simulation is run
rng(1235)
numModulationTypes = length(modulationTypes);
transDelay = 50;

%% Intialize the data storage location and file name format
%dataDirectory = fullfile('c:\',"ModClassDataFiles4_0GHz");%For fc=4 GHz
dataDirectory = fullfile('d:\',"ModulatonClassDataFiles70_0GHz");%For fc=26 GHz

disp("Data file directory is " + dataDirectory)
fileNameRoot = "csv";
disp("Generating data and saving in data files...")
[success,msg,msgID] = mkdir(dataDirectory);
if ~success
    error(msgID,msg)
end
numFramesPerModType = 1000;%number of frame per modulation

%% Main loop for parametric variation
for modType = 1:numModulationTypes
datamodu=zeros(spf,2,numFramesPerModType*length(antennaSpacing));

    for SpacingIndex=1:length(antennaSpacing)
SNRval=SNR(SpacingIndex);

channel = helperModClassTestChannel(...
    'SampleRate', fs, ...
    'SNR', SNRval, ...
    'PathDelays', [0 1.8 3.4] / fs, ...
    'AveragePathGains', [0 -2 -10], ...
    'KFactor', 4, ...
    'MaximumDopplerShift', 4, ...
    'MaximumClockOffset', 5, ...
    'CenterFrequency', fc);

label = modulationTypes(modType);
    numSymbols = (numFramesPerModType / sps);
    dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
    modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs);
 % Digital modulation types use a center frequency of 902 MHz
    channel.CenterFrequency = fc; 
    for p=1:numFramesPerModType
        % Generate random data
        x = dataSrc();%input vector

        % Modulate
        y = modulator(x);%after modulation

        % Pass through independent channels
        rxSamples = channel(y);%output data dimension 1024x numFramesPerModType X numModulationTypes

        %new data generation format
        numSamples = length(rxSamples);
        windowLength=spf;
        stepSize=spf;
        offset=transDelay;
        numFrames =floor(((numSamples-offset)-(windowLength-stepSize))/stepSize);
        frame= zeros([windowLength,numFrames],class(rxSamples));
        startIdx = offset + randi([0 sps]);
        frameCnt = 1;
        while startIdx + windowLength < numSamples
            xWindowed = rxSamples(startIdx+(0:windowLength-1),1);
            framePower = mean(abs(xWindowed).^2);
            xWindowed = xWindowed / sqrt(framePower);
            frame(:,frameCnt) = xWindowed;
            frameCnt = frameCnt + 1;
            startIdx = startIdx + stepSize;
        end
        datamodu_(:,SpacingIndex,p)=frame;
    end %end of each frame
    end %of each spacing
datamodu_=reshape(datamodu_,[spf length(antennaSpacing)*numFramesPerModType]);
datamodu(:,1,:)=real(datamodu_);
datamodu(:,2,:)=imag(datamodu_);
clearvars datamodu_ 

% Save data file
    fileName = fullfile(dataDirectory,...
        sprintf("%s",modulationTypes(modType)));
    save(fileName,"datamodu","label")

end % of each modulation
%% Related dependency functions
function src = helperModClassGetSource(modType, sps, spf, fs)
%helperModClassGetSource Source selector for modulation types
%    SRC = helperModClassGetSource(TYPE,SPS,SPF,FS) returns the data source
%    for the modulation type TYPE, with the number of samples per symbol
%    SPS, the number of samples per frame SPF, and the sampling frequency
%    FS.
%
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019 The MathWorks, Inc.

switch modType
    case {"BPSK","GFSK","CPFSK"}
        M = 2;
        src = @()randi([0 M-1],spf/sps,1);
    case {"QPSK","PAM4"}
        M = 4;
        src = @()randi([0 M-1],spf/sps,1);
    case "8PSK"
        M = 8;
        src = @()randi([0 M-1],spf/sps,1);
    case "16QAM"
        M = 16;
        src = @()randi([0 M-1],spf/sps,1);
    case "64QAM"
        M = 64;
        src = @()randi([0 M-1],spf/sps,1);
    case {"B-FM","DSB-AM","SSB-AM"}
        src = @()getAudio(spf,fs);
end
end

function x = getAudio(spf,fs)
%getAudio Audio source for analog modulation types
%    A = getAudio(SPF,FS) returns the audio source A, with the
%    number of samples per frame SPF, and the sample rate FS.

persistent audioSrc audioRC

if isempty(audioSrc)
    audioSrc = dsp.AudioFileReader('audio_mix_441.wav',...
        'SamplesPerFrame',spf,'PlayCount',inf);
    audioRC = dsp.SampleRateConverter('Bandwidth',30e3,...
        'InputSampleRate',audioSrc.SampleRate,...
        'OutputSampleRate',fs);
    [~,decimFactor] = getRateChangeFactors(audioRC);
    audioSrc.SamplesPerFrame = ceil(spf / fs * audioSrc.SampleRate / decimFactor) * decimFactor;
end

x = audioRC(audioSrc());
x = x(1:spf,1);
end
%%
function modulator = helperModClassGetModulator(modType, sps, fs)
%helperModClassGetModulator Modulation function selector
%   MOD = helperModClassGetModulator(TYPE,SPS,FS) returns the modulator
%   function handle MOD based on TYPE. SPS is the number of samples per
%   symbol and FS is the sample rate.
%
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019 The MathWorks, Inc.

switch modType
    case "BPSK"
        modulator = @(x)bpskModulator(x,sps);
    case "QPSK"
        modulator = @(x)qpskModulator(x,sps);
    case "8PSK"
        modulator = @(x)psk8Modulator(x,sps);
    case "16QAM"
        modulator = @(x)qam16Modulator(x,sps);
    case "64QAM"
        modulator = @(x)qam64Modulator(x,sps);
    case "GFSK"
        modulator = @(x)gfskModulator(x,sps);
    case "CPFSK"
        modulator = @(x)cpfskModulator(x,sps);
    case "PAM4"
        modulator = @(x)pam4Modulator(x,sps);
    case "B-FM"
        modulator = @(x)bfmModulator(x, fs);
    case "DSB-AM"
        modulator = @(x)dsbamModulator(x, fs);
    case "SSB-AM"
        modulator = @(x)ssbamModulator(x, fs);
end
end

function y = bpskModulator(x,sps)
%bpskModulator BPSK modulator with pulse shaping
%   Y = bpskModulator(X,SPS) BPSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 1]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
    filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,2);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qpskModulator(x,sps)
%qpskModulator QPSK modulator with pulse shaping
%   Y = qpskModulator(X,SPS) QPSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
    filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,4,pi/4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = psk8Modulator(x,sps)
%psk8Modulator 8-PSK modulator with pulse shaping
%   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 7]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
    filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,8);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam16Modulator(x,sps)
%qam16Modulator 16-QAM modulator with pulse shaping
%   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 15]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
    filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and pulse shape
syms = qammod(x,16,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam64Modulator(x,sps)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
    filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = qammod(x,64,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = pam4Modulator(x,sps)
%pam4Modulator PAM4 modulator with pulse shaping
%   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs amp
if isempty(filterCoeffs)
    filterCoeffs = rcosdesign(0.35, 4, sps);
    amp = 1 / sqrt(mean(abs(pammod(0:3, 4)).^2));
end
% Modulate
syms = amp * pammod(x,4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = gfskModulator(x,sps)
%gfskModulator GFSK modulator
%   Y = gfskModulator(X,SPS) GFSK modulates the input X and returns the
%   signal Y. X must be a column vector of values in the set [0 1]. The
%   BT product is 0.35 and the modulation index is 1. The output signal
%   Y has unit power.

persistent mod meanM
if isempty(mod)
    M = 2;
    mod = comm.CPMModulator(...
        'ModulationOrder', M, ...
        'FrequencyPulse', 'Gaussian', ...
        'BandwidthTimeProduct', 0.35, ...
        'ModulationIndex', 1, ...
        'SamplesPerSymbol', sps);
    meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end

function y = cpfskModulator(x,sps)
%cpfskModulator CPFSK modulator
%   Y = cpfskModulator(X,SPS) CPFSK modulates the input X and returns
%   the signal Y. X must be a column vector of values in the set [0 1].
%   the modulation index is 0.5. The output signal Y has unit power.

persistent mod meanM
if isempty(mod)
    M = 2;
    mod = comm.CPFSKModulator(...
        'ModulationOrder', M, ...
        'ModulationIndex', 0.5, ...
        'SamplesPerSymbol', sps);
    meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end

function y = bfmModulator(x,fs)
%bfmModulator Broadcast FM modulator
%   Y = bfmModulator(X,FS) broadcast FM modulates the input X and returns
%   the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The frequency deviation is 75 kHz
%   and the pre-emphasis filter time constant is 75 microseconds.

persistent mod
if isempty(mod)
    mod = comm.FMBroadcastModulator(...
        'AudioSampleRate', fs, ...
        'SampleRate', fs);
end
y = mod(x);
end

function y = dsbamModulator(x,fs)
%dsbamModulator Double sideband AM modulator
%   Y = dsbamModulator(X,FS) double sideband AM modulates the input X and
%   returns the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The IF frequency is 50 kHz.

y = ammod(x,50e3,fs);
end

function y = ssbamModulator(x,fs)
%ssbamModulator Single sideband AM modulator
%   Y = ssbamModulator(X,FS) single sideband AM modulates the input X and
%   returns the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The IF frequency is 50 kHz.

y = ssbmod(x,50e3,fs);
end
