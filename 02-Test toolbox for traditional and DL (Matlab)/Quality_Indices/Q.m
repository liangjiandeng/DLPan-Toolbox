%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Q/SSIM averaged on all bands.
% 
% Interface:
%           Q_avg = Q(I1,I2,L)
%
% Inputs:
%           I1:         First multispectral image;
%           I2:         Second multispectral image;
%           L:          Radiometric resolution.
%
% Outputs:
%           Q_avg:      Q index averaged on all bands.
% 
% References:
%           [Wang02]    Z. Wang and A. C. Bovik, “A universal image quality index,” IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81–84, March 2002.
%           [Vivone20]  G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                       IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Q_avg = Q(I1,I2,L)

Q_orig = zeros(1,size(I1,3));

for idim=1:size(I1,3),
%     Q_orig(idim) = ssim(I_GT(:,:,idim),I1U(:,:,idim), [0.01 0.03],fspecial('gaussian', 11, 1.5), L);
    Q_orig(idim) = img_qi(I1(:,:,idim),I2(:,:,idim), 32);
end

Q_avg = mean(Q_orig);

end