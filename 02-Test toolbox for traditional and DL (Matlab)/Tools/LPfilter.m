%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           LPfilter filters the panchromatic (PAN) image using à trous wavelet transform. 
% 
% Interface:
%           HRPanLP = LPfilter(HRPan,ratio)
%
% Inputs:
%           HRPan:          PAN image;
%           ratio:          Scale ratio between MS and PAN.
%
% Outputs:
%           HRPanLP:       Output filtered MS image.
% 
% References:
%           [Vivone15]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function HRPanLP = LPfilter(HRPan,ratio)

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

WT = ndwt2_working(HRPan,Levels,WF);

for ii = 2 : numel(WT.dec), WT.dec{ii} = zeros(size(WT.dec{ii})); end

HRPanLP = indwt2_working(WT,'c');

end