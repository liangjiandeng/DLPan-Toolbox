% I_MS:         Image to segment
% n_segm:       Number of segments
% Output:
% S:            Segmentation map.
function S = k_means_clustering(I_MS, n_segm)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+
%%%  k-means Segmentation of MS image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+

F1 = zeros(size(I_MS,1)*size(I_MS,2),size(I_MS,3));

for ibands = 1 :size(I_MS,3)
    a = I_MS(:,:,ibands);
    F1(:,ibands) = a(:)/max(a(:));
end
IDX = kmeans(F1,n_segm);
S = reshape(IDX,[size(I_MS,1) size(I_MS,2)]);

end
