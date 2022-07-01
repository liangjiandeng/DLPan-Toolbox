%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Full resolution quality indexes. 
% 
% Interface:
%           [D_lambda,D_S,QNR_index,SAM_index,sCC] = indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,tag,ratio)
%
% Inputs:
%           I_F:                Fused image;
%           I_MS_LR:            MS image;
%           I_PAN:              Panchromatic image;
%           L:                  Image radiometric resolution; 
%           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
%           I_MS:               MS image upsampled to the PAN size;
%           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
%           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
%           flagQNR:            if flagQNR == 1, the software uses the QNR otherwise the HQNR is used.
%
% Outputs:
%           D_lambda:           D_lambda index;
%           D_S:                D_S index;
%           QNR_index:          QNR index;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D_lambda,D_S,QNR_index] = indexes_evaluation_FS(I_F,I_MS_LR,I_PAN,L,th_values,I_MS,sensor,ratio,flagQNR)

if th_values
    I_F(I_F > 2^L) = 2^L;
    I_F(I_F < 0) = 0;
end

cd Quality_Indices

if flagQNR == 1
    [QNR_index,D_lambda,D_S]= QNR(I_F,I_MS,I_MS_LR,I_PAN,ratio);
else
    [QNR_index,D_lambda,D_S] = HQNR(I_F,I_MS_LR,I_MS,I_PAN,32,sensor,ratio);
end

cd ..

end