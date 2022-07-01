%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OMP_Rec_Detile performs:
% 1) The estimation of the coefficients \alpha at reduced resolution using an orthogonal matching pursuit (OMP) procedure for multispectral images;
% 2) The reconstruction of the patches at full resolution using the hypothesis of invariance among scales of the \alpha coefficients;
% 3) The detiling step to get the final image details at full resolution for the approach proposed in [Vicinanza15].
%
% INPUTS
%   Dl:                 Low spatial resolution dictionary (Low resolution PAN details) built as in [Vicinanza15];
%   Dh:                 High spatial resolution dictionary (PAN details) built as in [Vicinanza15];
%   ytilde_k:           Patches in column form of the details of the MS original image or the MS original image (depending on the flag "do_detail" in CSDetails);
%   H_PAN,L_PAN,C_PAN:  PAN (row and column) dimensions and number of MS spectral bands;
%   resize_fact:        Resize factor (ratio between PAN and MS images);
%   TS:                 Tiling (dimensions of the patches are TS x TS, e.g. 7 x 7);
%   ol:                 Overlap in pixels between contiguous tiles.
%   n_atoms:            max number of representation atoms
%
% OUTPUT
%   I_Fus_CS:           Reconstructed details (or fused image if do_detail flag is 0) using the CS approach in [Vicinanza15] for the final pansharpening product.
%
% REFERENCE
%   [Vicinanza15]       M.R. Vicinanza, et al. "A pansharpening method based on the sparse representation of injected details."
%                       IEEE Geoscience and Remote Sensing Letters 12.1 (2015): 180-184.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_CS = OMP_Rec_Detile(Dl, Dh, ytilde_k, H_PAN, L_PAN, C_MS, resize_fact, ol, TS, n_atoms)

I_Fus_CS = zeros ([H_PAN L_PAN C_MS]);
countpx = zeros ([H_PAN L_PAN C_MS]);
nr = ceil ((H_PAN/resize_fact - ol) / (TS - ol));
nc = ceil ((L_PAN/resize_fact - ol) / (TS - ol));
shiftr_glob = 0; shiftc_glob = 0;

if mod(H_PAN/resize_fact-ol, TS-ol) ~= 0
    shiftr_glob = TS-ol - mod (H_PAN/resize_fact-ol, TS-ol);
end

if mod(L_PAN/resize_fact-ol, TS-ol) ~= 0
    shiftc_glob = TS-ol - mod (L_PAN/resize_fact-ol, TS-ol);
end

alpha_count = 0;
Latom = size (Dl, 2);
Dict_Size = size (ytilde_k, 2);
iatom = 0;
for irow=1:nr
    for icol=1:nc
        iatom = iatom+1;
        if irow == nr
            shiftr = shiftr_glob;
        else
            shiftr = 0;
        end
        if icol == nc
            shiftc = shiftc_glob;
        else
            shiftc = 0;
        end
        blockr = ((irow-1)*(TS-ol)*resize_fact+1 - shiftr*resize_fact) : ((irow*TS-(irow-1)*ol)*resize_fact - shiftr*resize_fact);
        blockc = ((icol-1)*(TS-ol)*resize_fact+1 - shiftc*resize_fact) : ((icol*TS-(icol-1)*ol)*resize_fact - shiftc*resize_fact);
        Lr = length (blockr); Lc = length (blockc);
        y_cur = ytilde_k(:,iatom);
        
        % Sparse coding with OMP for MS data
        [alpha,inds] = OMP(Dl, y_cur, C_MS, iatom, n_atoms);

        % Patch reconstruction and detiling
        for iband = 1:C_MS
            reconstr_patch = Dh((iband-1)*TS^2*resize_fact^2+1:iband*TS^2*resize_fact^2,inds) * alpha(:,iband);
            I_Fus_CS(blockr,blockc,iband) = I_Fus_CS(blockr,blockc,iband) + reshape (reconstr_patch, Lr, Lc);
            countpx(blockr,blockc,iband) = countpx(blockr,blockc,iband) +1;
        end
        
        if mod(iatom,100)==1
            fprintf ('OMP band by band and detile: atom %i of %i\n', iatom, Dict_Size);
        end
        alpha_count = alpha_count + sum( sum(alpha,2)~=0 );
    end
end

% Average overlapping patches
I_Fus_CS = I_Fus_CS ./ countpx;

fprintf ('Sparsity di alfa = %.2f: %.1f atoms on %i used for each patch on average\n', (Dict_Size*Latom-alpha_count)/Dict_Size/Latom*100, alpha_count/Dict_Size, Dict_Size)

end