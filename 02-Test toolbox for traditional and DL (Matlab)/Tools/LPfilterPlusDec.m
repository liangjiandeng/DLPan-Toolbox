%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           LPfilterPlusDec filters and decimates the image I_PAN using a Starck and Murtagh (S&M) filter. 
% 
% Interface:
%           I_PAN_LR = LPfilterPlusDec(I_PAN,ratio)
%
% Inputs:
%           I_PAN:          Image to be filtered and decimated;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Resize factors power of 2.
%
% Outputs:
%           I_PAN_LR:       Filtered and decimated image.
% 
% References:
%           [Starck07]      J.-L. Starck, J. Fadili, and F. Murtagh, “The undecimated wavelet decomposition and its reconstruction,” IEEE Transactions on Image
%                           Processing, vol. 16, no. 2, pp. 297–309, February 2007.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I_PAN_LR = LPfilterPlusDec(I_PAN,ratio)

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

WT = ndwt2_working(I_PAN,Levels,WF);

for ii = 2 : numel(WT.dec), WT.dec{ii} = zeros(size(WT.dec{ii})); end

I_PAN_LR = indwt2_working(WT,'c');

I_PAN_LR = imresize(I_PAN_LR,1/ratio,'nearest');

end