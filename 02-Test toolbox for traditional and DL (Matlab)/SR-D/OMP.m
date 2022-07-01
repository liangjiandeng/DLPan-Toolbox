%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OMP is the Orthogonal matching Pursuit (OMP) modified to work with multispectral data.
% 
% INPUTS
%   D:                  Dictionary (matrix);
%   y:                  Measurements (column vector);
%   delta:              Maximum error allowed for the constraint y = D a;
%   nBands:             Number of MS spectral bands;
%   iatom:              Id of the actual atom under analysis.
%   n_atoms:            max number of representation atoms
%
% OUTPUTS
%   a:                  Estimated alphas;
%   indx:               Vector of the atom positions in the dictionary.
% 
% REFERENCE
%   [Vicinanza15]       M.R. Vicinanza, et al. "A pansharpening method based on the sparse representation of injected details." 
%                       IEEE Geoscience and Remote Sensing Letters 12.1 (2015): 180-184.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a, indx] = OMP(D, y,  nBands, iatom, n_atoms)


L_atom = size(D);
n = round(L_atom / nBands);
delta = 0;
res = y;
curr_delta = sum (res.^2);
j = 0;

while curr_delta > delta && j < n_atoms
    j = j+1;
    if j==1
        indx = iatom;
    else
        proj = D' * res;
        [~, imax] = max(abs(proj));
        imax = imax(1);
        indx = cat(2,indx,imax);
    end
    a = zeros (j, nBands);
    for iband = 1:nBands
        Di = D((iband-1)*n+1:iband*n,indx(1:j));
        yi = y((iband-1)*n+1:iband*n);
        DitDi = Di'*Di;
        if det (DitDi) > 1e-1
            a(:,iband) = ((DitDi)\(Di')) * yi;
        end
        Da((iband-1)*n+1:iband*n) = Di * a(:,iband);
    end
    res = y - Da';
    curr_delta = sum(res.^2);
end

end