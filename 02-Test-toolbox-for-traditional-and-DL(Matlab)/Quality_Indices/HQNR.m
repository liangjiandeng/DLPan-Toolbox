%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Hybrid Quality with No Reference (HQNR) index. 
% 
% Interface:
%           [HQNR_value,Dl,Ds] = HQNR(ps_ms,ms,msexp,pan,S,sensor,ratio)
%
% Inputs:
%           ps_ms:              Pansharpened image;
%           ms:                 Original MS image;
%           msexp:              MS image resampled to panchromatic scale;
%           pan:                Panchromatic image;
%           S:                  Block size (optional); Default value: 32;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
% 
% Outputs:
%           HQNR_value:          QNR index;
%           Dl:                  D_lambda index;
%           Ds:                  D_s index.
% 
% References:
%           [Alparone08]        L. Alparone, B. Aiazzi, S. Baronti, A. Garzelli, F. Nencini, and M. Selva, "Multispectral and panchromatic data fusion assessment without reference,"
%                               Photogrammetric Engineering and Remote Sensing, vol. 74, no. 2, pp. 193–200, February 2008. 
%           [Khan09]            M. M. Khan, L. Alparone, and J. Chanussot, "Pansharpening quality assessment using the modulation transfer functions of instruments", 
%                               IEEE Trans. Geosci. Remote Sens., vol. 11, no. 47, pp. 3880–3891, Nov. 2009.
%           [Aiazzi14]          B. Aiazzi, L. Alparone, S. Baronti, R. Carlà, A. Garzelli, and L. Santurri, 
%                               "Full scale assessment of pansharpening methods and data products", 
%                               in SPIE Remote Sensing, pp. 924 402 – 924 402, 2014.
%           [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                               IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [HQNR_value,Dl,Ds] = HQNR(ps_ms,ms,msexp,pan,S,sensor,ratio)

Dl = D_lambda_K(ps_ms,msexp,ratio,sensor,S);

Ds = D_s(ps_ms,msexp,ms,pan,ratio,S,1);

HQNR_value = (1-Dl)*(1-Ds);

end
