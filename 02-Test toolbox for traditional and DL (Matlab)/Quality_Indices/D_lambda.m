%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Quality with No Reference (QNR). Spectral distortion index. 
% 
% Interface:
%           D_lambda_index = D_lambda(I_F,I_MS,I_MS_LR,S,ratio,p)
%
% Inputs:
%           I_F:                Pansharpened image;
%           I_MS:               MS image resampled to panchromatic scale;
%           I_MS_LR:            Original MS image;
%           S:                  Block size (optional); Default value: 32;
%           ratio:              Resolution ratio;
%           p:                  Exponent value (optional); Default value: p = 1.
% 
% Outputs:
%           D_lambda_index:     D_lambda index.
% 
% References:
%           [Alparone08]        L. Alparone, B. Aiazzi, S. Baronti, A. Garzelli, F. Nencini, and M. Selva, "Multispectral and panchromatic data fusion assessment without reference,"
%                               Photogrammetric Engineering and Remote Sensing, vol. 74, no. 2, pp. 193–200, February 2008. 
%           [Vivone14]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D_lambda_index = D_lambda(I_F,I_MS,I_MS_LR,S,ratio,p)

flag_orig_paper = 0; % if 0, Toolbox 1.0, otherwise, original QNR paper 

if (size(I_F) ~= size(I_MS))
    error('The two input images must have the same dimensions')
end

[N,M,Nb] = size(I_F);

if (rem(N,S) ~= 0)
    error('The number of rows must be multiple of the block size')
end

if (rem(M,S) ~= 0)
    error('The number of columns must be multiple of the block size')
end

D_lambda_index = 0;
for i = 1:Nb-1
    for j = i+1:Nb 
        if flag_orig_paper == 0
            %%%%%%% Opt. 1 (as toolbox 1.0)
            band1 = I_MS(:,:,i);
            band2 = I_MS(:,:,j);
            fun_uqi = @(bs) uqi(bs.data,...
            band2(bs.location(1):bs.location(1)+S-1,...
            bs.location(2):bs.location(2)+S-1));
            Qmap_exp = blockproc(band1,[S S],fun_uqi);
        else
            %%%%%%% Opt. 2 (as paper QNR)
            band1 = I_MS_LR(:,:,i);
            band2 = I_MS_LR(:,:,j);
            fun_uqi = @(bs) uqi(bs.data,...
            band2(bs.location(1):bs.location(1)+S/ratio-1,...
            bs.location(2):bs.location(2)+S/ratio-1));
            Qmap_exp = blockproc(band1,[S/ratio S/ratio],fun_uqi);
        end
        Q_exp = mean2(Qmap_exp);
        
        band1 = I_F(:,:,i);
        band2 = I_F(:,:,j);
        fun_uqi = @(bs) uqi(bs.data,...
            band2(bs.location(1):bs.location(1)+S-1,...
            bs.location(2):bs.location(2)+S-1));
        Qmap_fused = blockproc(band1,[S S],fun_uqi);
        Q_fused = mean2(Qmap_fused);
        D_lambda_index = D_lambda_index + abs(Q_fused-Q_exp)^p;
    end
end
s = ((Nb^2)-Nb)/2;
D_lambda_index = (D_lambda_index/s)^(1/p);

end

%%%%%%% Q-index on x and y images
function Q = uqi(x,y)

x = double(x(:));
y = double(y(:));
mx = mean(x);
my = mean(y);
C = cov(x,y);

Q = 4 * C(1,2) * mx * my / (C(1,1)+C(2,2)) / (mx^2 + my^2);  

end