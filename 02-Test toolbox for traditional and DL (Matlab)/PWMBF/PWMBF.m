%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%       Model-based fusion using PCA and wavelets.
% 
% Interface:
%       Z = PWMBF(Pan,Low,ratio,r,wavelet,degrade,reduced,whiten)
% 
% Inputs:
%         Pan : Panchromatic image;
%          Low: Low spatial resolution MS image;
%        ratio: Scale ratio between Pan and Low;
%            r: Number of principal components;
%      wavelet: flag;
%      degrade: flag.
% 
% Output:
%    Z:     Pansharpened image;
% 
% References:
%           [Palsson15]     F. Palsson, J.R. Sveinsson, M.O. Ulfarsson, J.A. Benediktsson, "Model-based fusion of multi-and hyperspectral images using PCA and wavelets", 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2652-2663, May 2015.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Z = PWMBF(Pan,Low,ratio,r,wavelet,degrade)

addpath(sprintf('%s/rwt/bin',pwd))

% Wavelet parameters
L=4;
type='rwt';

Low=double(Low);
Pan=double(Pan);

N=size(Pan,1);
Q=size(Pan,3);
nb=size(Low,3);

if(r>nb)
    error('Number of PCs greater than number of bands');
end

X=Pan;
Ylow=Low;

if(degrade)
    X=imresize(Pan,1/ratio);
    Ylow=imresize(Low,1/ratio);
    N=N/4;
end

% Upsample Y
Y=imresize(Ylow,ratio,'bicubic');

% Degrade X
Xtilde=imresize(imresize(X,1/ratio,'bilinear'),ratio,'bicubic');

X=reshape(X,[N^2 Q]);
Xtilde=reshape(Xtilde,[N^2 Q]);
Y=reshape(Y,[N^2 nb]);

% PCA transform
[F, D, R]=svd(Y,'econ');
G=F*D;
U=R;

wfilter=daubcqf(4,'min');

if wavelet
    x=compute_PhiTX(Xtilde,L,wfilter,type);
    x0=compute_PhiTX(X,L,wfilter,type);
    y=compute_PhiTX(G(:,1:r),L,wfilter,type);
    yl=y(1:N^2,:);
    zh=zeros(3*L*N^2,r);
    for p=1:r
        for j=1:3*L
            xh=x(j*N^2+1:(j+1)*N^2,:);
            xh0=x0(j*N^2+1:(j+1)*N^2,:);
            yh=y(j*N^2+1:(j+1)*N^2,p);
            Cyy=yh'*yh/N^2;
            Cyx=yh'*xh/N^2;
            Cxx=xh'*xh/N^2;
            Cn=diag(mad(abs(yh))/0.6745).^2;
            inv_Cxx=inv(Cxx);
            Cy_x=Cyy-Cyx*inv_Cxx*Cyx';
            if Q>1
                CyxiCxx=Cyx*inv_Cxx;
                mu_zx=xh*CyxiCxx';
                mu_zx0=xh0*CyxiCxx';
            else
                mu_zx=repmat((Cyx*inv_Cxx)',[N^2 1]).*xh;
                mu_zx0=repmat((Cyx*inv_Cxx)',[N^2 1]).*xh0;
            end
            ymu=yh-mu_zx;
            CC=Cy_x*inv(Cy_x+Cn);
            zh((j-1)*N^2+1:N^2+(j-1)*N^2,p)=mu_zx0+ymu*CC;
        end
    end
    z=[yl;zh];
    B=compute_PhiX(z,L,wfilter,type);
    deg=0;
    if deg == 1
        U = U(:,1:r);
        Zhat=B*U';
    else
        G(:,1:r)=B;
        Zhat=G*U';
    end
else
    Cn=0;
    yh=G(:,1:r);
    xh=Xtilde;
    xh0=X;
    Cyy=yh'*yh/N^2;
    Cyx=yh'*xh/N^2;
    Cxx=xh'*xh/N^2;
    inv_Cxx=inv(Cxx);
    Cy_x=Cyy-Cyx*inv_Cxx*Cyx';
    CyxiCxx=Cyx*inv_Cxx;
    mu_zx=xh*CyxiCxx';
    mu_zx0=xh0*CyxiCxx';
    ymu=yh-mu_zx;
    CC=Cy_x/(Cy_x+Cn);
    B=mu_zx0+ymu*CC;
    G(:,1:r)=B;
    Zhat=G*U';
end

Z=reshape(Zhat,[N N nb]);

end