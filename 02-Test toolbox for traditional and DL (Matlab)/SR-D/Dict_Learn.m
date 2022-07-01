%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dict_Learn is the dictionary learning method for the 
% compressive sensing approach for Pansharpening proposed in [Vicinanza15].
% 
% INPUTS
%   I_PAN_D:            Details of the panchromatic image;
%   I_PAN_LR_D:         Details of the low resolution panchromatic image;
%   I_MS_LR_D:          Details of the MS original image or the MS original image (depending on the flag "do_detail" in CSDetails);
%   resize_fact:        Resize factor (ratio between PAN and MS images);
%   TS:                 Tiling (dimensions of the patches are TS x TS, e.g. 7 x 7);
%   ol:                 Overlap in pixels between contiguous tiles.
% 
% OUTPUTS
%   Dh:                 High spatial resolution dictionary (PAN details) built as in [Vicinanza15]; 
%   Dl:                 Low spatial resolution dictionary (Low resolution PAN details) built as in [Vicinanza15];
%   ytilde_k:           Patches in column form of the details of the MS original image or the MS original image (depending on the flag "do_detail" in CSDetails).
% 
% REFERENCE
%   [Vicinanza15]       M.R. Vicinanza, et al. "A pansharpening method based on the sparse representation of injected details." 
%                       IEEE Geoscience and Remote Sensing Letters 12.1 (2015): 180-184.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Dh, Dl, ytilde_k] = Dict_Learn(I_PAN_D, I_PAN_LR_D, I_MS_LR_D, resize_fact, TS, ol)

nr = ceil ((size(I_PAN_D,1)/resize_fact - ol) / (TS - ol));
nc = ceil ((size(I_PAN_D,2)/resize_fact - ol) / (TS - ol));
nBands = size (I_MS_LR_D,3);

Dh = zeros (TS^2*resize_fact^2*nBands, nr*nc);
Dl = zeros (TS^2*nBands, nr*nc);
ytilde_k = zeros (TS^2*nBands, nr*nc);

% Building the dictionaries (Dh and Dl)
icount = 0;
for irow=1:nr
    for icol=1:nc
        icount = icount + 1;
        shiftr = 0; shiftc = 0;
        if irow == nr && mod(size(I_MS_LR_D,1)-ol, TS-ol) ~= 0
            shiftr = TS-ol - mod (size(I_MS_LR_D,1)-ol, TS-ol);
        end
        if icol == nc && mod(size(I_MS_LR_D,2)-ol, TS-ol) ~= 0
            shiftc = TS-ol - mod (size(I_MS_LR_D,2)-ol, TS-ol);
        end
        blockr = ((irow-1)*(TS-ol)*resize_fact+1 - shiftr*resize_fact) : ((irow*TS-(irow-1)*ol)*resize_fact - shiftr*resize_fact);
        blockc = ((icol-1)*(TS-ol)*resize_fact+1 - shiftc*resize_fact) : ((icol*TS-(icol-1)*ol)*resize_fact - shiftc*resize_fact);

        blockrl = ((irow-1)*(TS-ol)+1 - shiftr) : (irow*TS-(irow-1)*ol - shiftr);
        blockcl = ((icol-1)*(TS-ol)+1 - shiftc) : (icol*TS-(icol-1)*ol - shiftc);

        for iband = 1:nBands          
            colmn = I_PAN_D(blockr,blockc,iband);
            colmnlr = I_PAN_LR_D(blockrl,blockcl,iband);
            colmny = I_MS_LR_D(blockrl,blockcl,iband);
            Dh((iband-1)*TS^2*resize_fact^2+1:(iband-1)*TS^2*resize_fact^2+length(colmn(:)),icount) = (colmn(:));
            Dl((iband-1)*TS^2+1:(iband-1)*TS^2+length(colmnlr(:)),icount) = (colmnlr(:));
            ytilde_k((iband-1)*TS^2+1:(iband-1)*TS^2+length(colmny(:)),icount) = (colmny(:));
        end
    end
end

end