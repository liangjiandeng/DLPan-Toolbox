%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           This function minimizes 
%               J(x) = || y - M*x ||^2 + lambda*TV(x)
%           where
%               y = [yms^T, ypan^T]^T
%               x is the pansharpened ms image
%               M models the relationship between
%               y and x; see [Palsson07] for details
% 
% Interface:
%           x = TV_pansharpen(yms,ypan,alpha,lambda,c,maxiter,w)
%
% Inputs:
%           yms:            The observed MS image;
%           ypan:           The PAN image;
%           alpha:          convergence parameter 1, suggested value=0.75;
%               c:          convergence parameter 2, suggested value=8;
%         maxiter:          number of iterations;
%               w:          We assume the pan image to be a linear 
%                           combination of the pansharpened ms image,
%                           w contains the weights.                    
% Output:
%               x:          Pansharpened image.
% 
% Reference:
%           [Palsson14]     F. Palsson, J.R. Sveinsson, and M.O. Ulfarsson, “A New Pansharpening Algorithm Based on Total Variation”
%                           IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 1, pp. 318 - 322, 2014.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = TV_pansharpen(yms,ypan,alpha,lambda,c,maxiter,w)
    
    z=zeros([size(ypan) size(yms,3)*2]);
    x=zeros([size(ypan) size(yms,3)]);

    for k=1:maxiter
        b=computeb(yms,ypan,x,alpha,w);
        z=znext(z,x,b,alpha,lambda,c);
        x=xnext(z,b,alpha);
    end
end

function b=computeb(yms,ypan,xk,alpha,w)
    [Hxms, Hxpan]=computeH(xk,w);
    b=alpha*xk+adjointH(yms-Hxms,ypan-Hxpan,w);
end

function [yms, ypan]=computeH(x,w)

    ypan=zeros([size(x,1) size(x,2)]);

    for i=1:size(x,3)
        yms(:,:,i)=decimate(x(:,:,i));
        ypan=ypan+w(i)*x(:,:,i);
    end
end

function y=decimate(x)
    % y = imfilter(x,fspecial('Gaussian',9,sigma),'replicate');
    % y = imfilter(y,fspecial('average',4),'replicate');
    % y = y(1:4:end,1:4:end);
    %  h=0.25*[1 1 1 1];
    %  x=imfilter(x,h'*h,'symmetric','same');
    %  y=downsample(downsample(x,4,1)',4,1)';
    y=imresize(x,0.25,'bilinear');
    % y=MTF_downsample(x,'QB','none',4,1);
    % y=imresize(imresize(x,1/4,'bicubic'),4,'bicubic');
end
     
function x=adjointH(yms,ypan,w)
    for i=1:size(yms,3)
        x(:,:,i)=interpolate(yms(:,:,i))+w(i)*ypan;
    end
end

function y=interpolate(x)
    % y = upsample(upsample(x,4)',4)';
    y=imresize(x,4,'bilinear');
    % y = imfilter(y,fspecial('Gaussian',9,sigma),'replicate');
    % y = imfilter(y,fspecial('average',4),'replicate');
    %  y=imresize(x,4,'bicubic');
    % y=MTF_upsample(x,'IKONOS','none',4,1);
    % y=interp23tap(x,4);
end

function z1=znext(z0,x0,b,alpha,lambda,c)
    for i=1:size(x0,3)
        W(:,:,i)= 2* alpha/lambda * sqrt(Dx(x0(:,:,i)).^2+Dy(x0(:,:,i)).^2)+c;
        W(:,:,i+size(x0,3))=2 * alpha/lambda * sqrt(Dx(x0(:,:,i)).^2+Dy(x0(:,:,i)).^2)+c;
    end
    z1=(computeDb(b)+cIDDTz(z0,c))./W;
end

function DX = Dx(v) 
    DX=[diff(v,1,2) zeros(size(v,1),1)];
end

function DY = Dy(v) 
    DY=[diff(v); zeros(1,size(v,2))];
end

function Db=computeDb(b)

    for i=1:size(b,3)
        Db(:,:,i)=Dx(b(:,:,i));
    end
    for i=size(b,3)+1:2*size(b,3)
        Db(:,:,i)=Dy(b(:,:,i-size(b,3)));
    end
end

function ddtz=cIDDTz(z,c)

    for i=1:size(z,3)/2
        dtz(:,:,i)=DxT(z(:,:,i))+DyT(z(:,:,i+4));
    end

    ddtz=computeDb(dtz);
    cIddtz=c*z-ddtz;
end

function DXT=DxT(v)
    DXT=DyT(v')';
end

function DYT = DyT(v)

    u0 = -v(1,:);
    u1 = -diff(v);
    u2 = v(end-1,:);
    DYT = [u0; u1(1:(end-1),:); u2];
    return
end

function x1=xnext(z1,b,alpha)
    x1=(b-DTz(z1))./alpha;
end

function dtz=DTz(z)
    for i=1:size(z,3)/2
        dtz(:,:,i)=DxT(z(:,:,i))+DyT(z(:,:,i+4));
    end
end