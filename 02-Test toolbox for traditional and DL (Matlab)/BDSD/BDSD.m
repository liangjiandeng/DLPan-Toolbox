%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           BDSD fuses the upsampled MultiSpectral (MS) and PANchromatic (PAN) images by 
%           exploiting the Band-Dependent Spatial-Detail (BDSD) algorithm. 
% 
% Interface:
%           I_Fus_BDSD = BDSD(I_MS,I_PAN,ratio,S,sensor)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           S:              Local estimation on SxS distinct blocks (typically 128x128); 
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS').
%
% Output:
%           I_Fus_BDSD:     BDSD pansharpened image.
% 
% References:
%           [Garzelli08]    A. Garzelli, F. Nencini, and L. Capobianco, “Optimal MMSE pan sharpening of very high resolution multispectral images,” 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 46, no. 1, pp. 228–236, January 2008.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
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
function I_Fus_BDSD = BDSD(I_MS,I_PAN,ratio,S,sensor)

%%%
% Control of input parameters and initialization
%%%
if (S > 1)
    if(rem(S,2) && S >1)
        fprintf(1,'\n\n ');
        error('block size for local estimation must be even')
    end

    if(rem(S,ratio))
        fprintf(1,'\n\n ');
        error('block size must be multiple of ratio')
    end

    [N,M] = size(I_PAN);

    if(rem(N,S)||rem(M,S))
        fprintf(1,'\n\n ');
        error('x and y dims of pan must be multiple of the block size')
    end
end

I_MS = double(I_MS);
I_PAN = double(I_PAN);

%%%
% Reduced resolution
%%%

pan_LP = MTF_PAN(I_PAN,sensor,ratio);
pan_LP_d = pan_LP(3:ratio:end,3:ratio:end);

ms_orig = imresize(I_MS,1/ratio);

ms_LP_d = MTF(ms_orig,sensor,ratio);

%%%
% Parameter estimation at reduced resolution
%%%
in3 = cat(3,ms_LP_d,ms_orig,pan_LP_d);
fun_eg = @(bs) estimate_gamma_cube(bs.data,S,ratio);
gamma = blockproc(in3,[S/ratio S/ratio],fun_eg);

%%%
% Fusion
%%%
in3 = cat(3,I_MS,I_PAN,gamma);
fun_Hi = @(bs) compH_inject(bs.data,S);

I_Fus_BDSD = blockproc(in3,[S S],fun_Hi);

%%%_______________________________________________________________
%%%
function gamma = estimate_gamma_cube(in3,S,ratio)
Nb = (size(in3,3)-1)/2;
hs_LP_d = in3(:,:,1:Nb);
hs_orig = in3(:,:,Nb+1:2*Nb);
pan_LP_d = in3(:,:,2*Nb+1);
% Compute Hd
Hd = zeros(S*S/ratio/ratio,Nb+1);
for k=1:Nb
    b = hs_LP_d(:,:,k);
    Hd(:,k) = b(:);
end
Hd(:,Nb+1) = pan_LP_d(:);
% Estimate gamma
B = (Hd'*Hd)\Hd';
gamma = zeros(Nb+1,Nb);
for k=1:Nb
    b = hs_orig(:,:,k);
    bd = hs_LP_d(:,:,k);
    gamma(:,k) = B *(b(:)-bd(:));
end
gamma = padarray(gamma,[S-Nb-1 S-Nb],0,'post');



%%%_______________________________________________________________
%%%
function ms_en = compH_inject(in3,S)
Nb = size(in3,3)-2;
hs = in3(:,:,1:Nb);
pan = in3(:,:,Nb+1);
gamma = in3(:,:,Nb+2); 
% Compute H
[N,M,Nb] = size(hs);
H = zeros(S*S,Nb+1);
for k=1:Nb
    b = hs(:,:,k);
    H(:,k) = b(:);
end
H(:,Nb+1) = pan(:);
% Inject
g = gamma(1:Nb+1,1:Nb);
ms_en = zeros(N,M,Nb);
for k=1:Nb
    b = hs(:,:,k);
    b_en = b(:) + H * g(:,k);
    ms_en(:,:,k) = reshape(b_en,N,M);
end