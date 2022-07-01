%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Quality with No Reference (QNR) index. 
% 
% Interface:
%           [QNR_index,D_lambda_index,D_s_index] = QNR(I_F,I_MS,I_MS_LR,I_PAN,ratio,S,p,q,alpha,beta)
%
% Inputs:
%           I_F:                Pansharpened image;
%           I_MS:               MS image resampled to panchromatic scale;
%           I_MS_LR:            Original MS image;
%           I_PAN:              Panchromatic image;
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           S:                  Block size (optional); Default value: 32;
%           p, q, alpha, beta:  Exponent values (optional); Default values: p = q = alpha = beta = 1.
% 
% Outputs:
%           QNR_index:          QNR index;
%           D_lambda_index:     D_lambda index;
%           D_s_index:          D_s index.
% 
% References:
%           [Alparone08]        L. Alparone, B. Aiazzi, S. Baronti, A. Garzelli, F. Nencini, and M. Selva, "Multispectral and panchromatic data fusion assessment without reference,"
%                               Photogrammetric Engineering and Remote Sensing, vol. 74, no. 2, pp. 193–200, February 2008. 
%           [Vivone15]          G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                               IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%           [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                               IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [QNR_index,D_lambda_index,D_s_index] = QNR(I_F,I_MS,I_MS_LR,I_PAN,ratio,S,p,q,alpha,beta)

if nargin < 11, beta=1; end
if nargin < 10, alpha=1; end
if nargin < 9, q=1; end
if nargin < 8, p=1; end
if nargin < 7, S=32; end

D_lambda_index = D_lambda(I_F,I_MS,I_MS_LR,S,ratio,p);

D_s_index = D_s(I_F,I_MS,I_MS_LR,I_PAN,ratio,S,q);

QNR_index = (1-D_lambda_index)^alpha * (1-D_s_index)^beta;

end