% Description:
%           gen_LP_image generates the Low Resolution version of the PAN image required for the calculation of the
%           segmentation-based version of the Gram-Schmidt algorithm, based on the segmentation S.
%
% Interface:
%           I_LR_input = gen_LP_image(Local_algorithm,I_MS,I_PAN,I_MS_LR,ratio,sensor,S)
%
% Inputs:
%           PS_algorithm: Employed segmentation-based algorithm
%                            ('GSA','GS2GLP')
%           I_MS:            MS image upsampled at PAN scale
%           I_PAN:           PAN image
%           I_MS_LR:         MS image
%           ratio:           Scale ratio between MS and PAN. Pre-condition: Integer value.
%           sensor:          String for type of sensor (e.g. 'WV2','IKONOS');
%
% Outputs:
%           I_LR_input:  Low Resolution  version of the PAN image
%
% References:
%
%           [Restaino17] R. Restaino, M. Dalla Mura, G. Vivone, J. Chanussot, “Context-Adaptive Pansharpening Based on Image Segmentation”,
%                        IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 2, pp. 753–766, February 2017.
%           [Vivone15]   G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”,
%                        IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_LR_input = gen_LP_image(PS_algorithm,I_MS,I_PAN,I_MS_LR,ratio,sensor)

switch PS_algorithm
        
    case 'GSA'
        %%%%%%%%% Generation of LR PAN image
        PAN_LP = LPfilterGauss(I_PAN,ratio);
        %%%%%%%%%% Estimation of weights
        PAN_LP2 = imresize(PAN_LP,1/ratio,'nearest');
        alpha= estimation_alpha(cat(3,I_MS_LR,ones(size(I_MS_LR,1),size(I_MS_LR,2))),PAN_LP2,'global');
        [Height,Width,Bands] = size(I_MS);
        I_MS_col = reshape(double(I_MS), Height*Width, Bands);
        alpha = repmat(alpha', [size(I_MS_col,1),1]);
        I_LR_col = sum([I_MS_col, ones(size(I_MS_col,1),1)] .* alpha, 2);
        I_LR_input = reshape(I_LR_col, Height, Width);
        
    case 'GS2GLP'
        h = genMTF(ratio, sensor, size(I_MS,3));
        for ii=1:size(h, 3)
            PAN_LP(:,:,ii) = imfilter(I_PAN,real(h(:,:,ii)),'replicate');
        end
        PAN_LP2 = imresize(PAN_LP,1/ratio,'nearest');
        I_LR_input = interp23tap(PAN_LP2,ratio);
end