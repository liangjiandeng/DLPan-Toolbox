%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           AWLP fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Additive Wavelet Luminance Proportional (AWLP) algorithm.
% 
% Interface:
%           I_Fus_AWLP = AWLP(I_MS,I_PAN,ratio)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image;
%           ratio:      Scale ratio between MS and PAN. Pre-condition: Integer value.
%
% Outputs:
%           I_Fus_AWLP: AWLP pasharpened image.
% 
% References:
%           [Otazu05]       X. Otazu, M. Gonzalez-Audcana, O. Fors, and J. Nunez, “Introduction of sensor spectral response into image fusion methods.
%                           Application to wavelet-based methods,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 10, pp. 2376–2385,
%                           October 2005.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%           [Alparone17]    L. Alparone, A. Garzelli, and G. Vivone, "Intersensor statistical matching for pansharpening: Theoretical issues and practical solutions",
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 8, pp. 4682-4695, 2017.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2019
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_AWLP = AWLP(I_MS,I_PAN,ratio)

[Height,Width,Bands]=size(I_MS);
I_Fus_AWLP=zeros(Height,Width,Bands,'double');

SumImage=sum(I_MS,3)/Bands;

IntensityRatio = zeros(size(I_MS));
for i=1:Bands
    IntensityRatio(:,:,i)=I_MS(:,:,i)./(SumImage+eps);
end

I_PAN = repmat(I_PAN,[1 1 size(I_MS,3)]);

% for ii = 1 : size(I_MS,3)    
%   I_PAN(:,:,ii) = (I_PAN(:,:,ii) - mean2(I_PAN(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(I_PAN(:,:,ii))) + mean2(I_MS(:,:,ii));  
% end
imageHR_LR=imresize(imresize(I_PAN,1/ratio),ratio);
for ii = 1 : size(I_MS,3)
    I_PAN(:,:,ii) = (I_PAN(:,:,ii) - mean2(I_PAN(:,:,ii))).*(std2(I_MS(:,:,ii))./std2(imageHR_LR(:,:,ii))) + mean2(I_MS(:,:,ii));
end

h=[1 4 6 4 1 ]/16;
g=[0 0 1 0 0 ]-h;
htilde=[ 1 4 6 4 1]/16;
gtilde=[ 0 0 1 0 0 ]+htilde;
h=sqrt(2)*h;
g=sqrt(2)*g;
htilde=sqrt(2)*htilde;
gtilde=sqrt(2)*gtilde;
WF={h,g,htilde,gtilde};

Levels = ceil(log2(ratio));

for i=1:Bands
    WT = ndwt2_working(I_PAN(:,:,i),Levels,WF);    
    for ii = 2 : numel(WT.dec), WT.dec{ii} = zeros(size(WT.dec{ii})); end
    StepDetails = I_PAN(:,:,i) - indwt2_working(WT,'c');
%%%%%%%%% OLD [as in the article Otazu05]
%     sINI = WT.sizeINI;
%     
%     StepDetails = zeros(sINI);
%     
%     for ii = 2 : numel(WT.dec)
%         h = WT.dec{ii};
%         h = imcrop(h,[(size(h,1) - sINI(1))/2 + 1,(size(h,2) - sINI(2))/2 + 1, sINI(1) - 1, sINI(2) - 1]);
%         StepDetails = StepDetails + h; 
%     end
%%%%%%%%%%%%%%%%%%%
    I_Fus_AWLP(:,:,i) = StepDetails .* IntensityRatio(:,:,i)+I_MS(:,:,i);
end

end