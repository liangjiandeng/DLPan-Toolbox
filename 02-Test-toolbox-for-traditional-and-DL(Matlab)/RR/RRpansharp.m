%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%     This method performs pansharpening. We assume that
%     the noisy satellite images yi, i=1,...,L, where y1 is the PAN image
%     and yi, i=2,...,L are the observed MS images, are related to the full
%     resolution target images by
% 
%       yi = Mi*Bi*xi + ni, i=1,...,L 
% 
%     where Mi is a downsampling operator, Bi is a circulant blurring matrix,
%     and ni is noise.  The method solves
%            min (1/2) sum_{i=1}^L || y_i - Mi*Bi*G*fi ||^2  + lambda * phi(G)
%            F, G
%     where phi is a regularizer function.
%     The function returns Xhat=G*F'. See [1] and [2] for details.
% 
% Interface:
%       Xhat_im = RRpansharp(Yim,varargin)
% 
% Inputs:
%         Yim : 1xL cell array containing the observed images the first image
%               is the PAN image and the last L-1 images are the MS images;
%       CDiter: Number of cyclic descent iterations. 
%               CDiter=100 is the default;
%            r: The subspace dimension;
%       lambda: The regularization parameter, lambda=0.005 is the 
%               default;
%            q: penalty weights;
%           X0: Initial value for X = G * F'.
% 
% Outputs:
%    Xhat_im: estimated image (3D) at high resolution for each 
%             spectral channel.
% 
% References:
%           [Ulfarsson19]   M.O. Ulfarsson, F. Palsson, M.Dalla Mura, J.R. Sveinsson, "Sentinel-2 Sharpening using a Reduced-Rank Method", 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 9, pp. 6408-6420, 2019.
%           [Palsson19]     F. Palsson, MO. Ulfarsson, and JR. Sveinsson, "Model-Based Reduced-Rank Pansharpening", 
%                           IEEE Geoscience and Remote Sensing Letters, 2019
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Xhat_im = RRpansharp(Yim,varargin)

    % import the manopt optimizer
    addpath('./manopt')
    p1=pwd;
    cd('./manopt');
    importmanopt
    cd(p1)
    % initialization
    CDiter=10;
    r=7;
    lambda=0.005;
    X0 = '';
    tolgradnorm = 0.1;
    if(r==7)
        q = [1, 1.5, 4, 8, 15, 15, 20 ]';
    else
        q = ones(r,1);
    end
    Gstep_only=0;
    GCV = 0;
    for i=1:2:(length(varargin)-1)
        switch varargin{i}
            case 'CDiter'
                CDiter=varargin{i+1};
            case 'r'
                r=varargin{i+1};
            case 'lambda'
                lambda=varargin{i+1};
            case 'q'
                q=varargin{i+1};
            case 'X0'
                X0 = varargin{i+1};
            case 'tolgradnorm'
                tolgradnorm = varargin{i+1};
            case 'Gstep_only'
                Gstep_only = varargin{i+1};
            case 'GCV'
                GCV = varargin{i+1};
            case 'd'
                d = varargin{i+1};
            case 'mtf'
                mtf = varargin{i+1};
        end
    end
    tic;
    if(length(q)~=r), error('The length of q has to match r'); end
    % dimensions of the inputs
    L=length(Yim);
    for i=1:L, Yim{i}=double(Yim{i}); end
    [nl,nc] = size(Yim{1});
    n = nl*nc;
    [Yim2, av] = normaliseData(Yim);
    % Sequence of bands
    % [B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12]
    % subsampling factors (in pixels)
    %d = [6 1 1 1 2 2 2 1 2 6 2 2]';
    % convolution  operators (Gaussian convolution filters), taken from ref [5]
    %mtf = [ .32 .26 .28 .24 .38 .34 .34 .26 .33 .26 .22 .23];
    sdf = d.*sqrt(-2*log(mtf)/pi^2)';
    % Do not sharpen high-res bands
    sdf(d==1) = 0;
    % remove border for computing the subspace and the result (because of
    % circular assumption
    limsub = 2;
    % kernel filter support
    dx = 12;
    dy = 12;
    % Define blurring operators
    FBM = createConvKernel(sdf,d,nl,nc,L,dx,dy);
    % IMPORTANT!!!
    % Note that the blur kernels are shifted to accomodate the co-registration
    % of real images with different resolutions.
    [Y,M,F]=initialization(Yim2,sdf,nl,nc,L,dx,dy,d,limsub,r);
    Mask=reshape(M,[n,L])';
    % CD
    if isempty(X0)
        Z = zeros(r,n); 
    else
        [X0, ~] = normaliseData(X0);
        X0 = reshape(X0,[n,L])';
        [F,D,V]=svd(X0,'econ');
        F = F(:,1:r);
        Z = D(1:r,1:r)*V(:,1:r)';
    end
    % Operators for differences
    [FDH,FDV,FDHC,FDVC] = createDiffkernels(nl,nc,r);
    % Compute weights
    sigmas = 1;
    W = computeWeights(Y,d,sigmas,nl);
    Whalf=W.^(1/2);
    if( GCV == 1), Gstep_only=1; end
    if( Gstep_only ~= 0), CDiter=1; end
    for jCD=1:CDiter
       [Z,Jcost(jCD),options]=Zstep(Y,FBM,F,lambda,nl,nc,Z,Mask,q,FDH,FDV,FDHC,FDVC,W,Whalf,tolgradnorm);              
       if(Gstep_only==0) 
           F1=Fstep(F,Z,Y,FBM,nl,nc,Mask);  
           F=F1;
       end
       if( GCV==1 )
            Ynoise = ( abs(Y) > 0 ) .* randn( size(Y) );
            [Znoise]=Zstep(Ynoise,FBM,F,lambda,nl,nc,Z,Mask,q,FDH,FDV,FDHC,FDVC,W,Whalf,tolgradnorm);
            HtHBXnoise = Mask.*ConvCM(F*Znoise,FBM,nl);
            Ynoise = Ynoise([2:end],:); 
            HtHBXnoise = HtHBXnoise([2:end],:);
            den = trace(Ynoise*(Ynoise - HtHBXnoise)');           
            HtHBX=Mask.*ConvCM(F*Z,FBM,nl); 
            num = norm( Y([2:end],:) - HtHBX([2:end],:) , 'fro')^2;         
       end
    end
    
    Xhat_im = conv2im(F*Z,nl,nc,L);
    Xhat_im = unnormaliseData(Xhat_im,av);
    Xhat_im = Xhat_im(:,:,2:end);
end

function [Y,M,F]=initialization(Yim2,sdf,nl,nc,L,dx,dy,d,limsub,r)
    FBM2 = createConvKernelSubspace(sdf,nl,nc,L,dx,dy);
    % Generate LR MS image FOR SUBSPACE
    % Upsample image via interpolation
    for i=1:L
        Ylim(:,:,i) = imresize(Yim2{i},d(i));
    end
    Y2im=real(ifft2(fft2(Ylim).*FBM2));
    Y2tr=Y2im(limsub+1:end-limsub,limsub+1:end-limsub,:);
    Y2n = reshape(Y2tr,[(nl-4)*(nc-4),L]); 
    % SVD analysis
    % Y2n is the image for subspace with the removed border
    [F,D,P] = svd(Y2n','econ');
    F=F(:,1:r);
    [M, Y] = createSubsampling(Yim2,d,nl,nc,L);
end


function [Z, xcost,options]=Zstep(Y,FBM,F,tau,nl,nc,Z,Mask,q,FDH,FDV,FDHC,FDVC,W,Whalf,tolgradnorm)
    r = size(F,2);
    n = nl*nc;     
    UBTMTy=F'*ConvCM(Y,conj(FBM),nl); 
    [Z] = CG(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W);
    xcost=1;
    options=[];    
end      

function F1=Fstep(F,Z,Y,FBM,nl,nc,Mask)
     F0=F;%   U; % initialization
     BTXhat =  ConvCM(F0*Z,FBM,nl);
     MBTXhat=Mask.*BTXhat;
     [L,r]=size(F);
     for ii=1:L
        MBZT(:,:,ii)=repmat(Mask(ii,:),[r,1]).*ConvCM(Z,repmat(FBM(:,:,ii),[1,1,r]),nl);
        A(:,:,ii)=MBZT(:,:,ii)*MBZT(:,:,ii)';
        ZBMTy(:,ii)=MBZT(:,:,ii)*Y(ii,:)';
     end
     ZBYT=ZBMTy';%    BTY*Z';
     manifold = stiefelfactory(L,r,1); %euclideanfactory(L,r); 
     problem.M = manifold;
     problem.cost  = @(F) costF(F,MBZT,Y); 
     problem.egrad = @(F) egrad(F,A,ZBYT);  
     warning('off', 'manopt:getHessian:approx') 
     options.tolgradnorm = 1e-2;
     options.verbosity=0;
     [F1, xcost, info, options] = trustregions(problem,F0,options);

end

% Cost functions

function [Ju]=costF(F,MBZT,Y)
    L=size(F,1);
    Ju=0;
    for i=1:L
        fi=F(i,:)';
        yi=Y(i,:)';
        Ju=Ju+0.5*norm(MBZT(:,:,i)'*fi-yi,'fro')^2;
    end
end

function [Du]=egrad(F,A,ZBYT)
    p=size(A,3);
    Du=0*F;
    for ii=1:p
        Du(ii,:)=F(ii,:)*A(:,:,ii)'-ZBYT(ii,:);
    end
end


%%% AUXILILARY FUNCTIONS

function [FDH,FDV,FDHC,FDVC] = createDiffkernels(nl,nc,r)
    dh = zeros(nl,nc);
    dh(1,1) = 1;
    dh(1,nc) = -1;
    dv = zeros(nl,nc);
    dv(1,1) = 1;
    dv(nl,1) = -1;
    FDH = repmat(fft2(dh),1,1,r);
    FDV = repmat(fft2(dv),1,1,r);
    FDHC = conj(FDH);
    FDVC = conj(FDV);
end


function [Yim, av] = normaliseData(Yim)
    % Normalize each cell to unit power
    if iscell(Yim)
        % mean squared power = 1
        nb = length(Yim);
        for i=1:nb
            av(i,1) = mean2(Yim{i}.^2);
            Yim{i,1} = sqrt(Yim{i}.^2/av(i,1));
        end   
    else
        nb = size(Yim,3);
        for i=1:nb
            av(i,1) = mean2(Yim(:,:,i).^2);
            Yim(:,:,i) = sqrt(Yim(:,:,i).^2/av(i,1));
        end
    end
end

function FBM = createConvKernel(sdf,d,nl,nc,L,dx,dy)
    %--------------------------------------------------------------------------
    %   Build convolution kernels
    %--------------------------------------------------------------------------
    
    middlel=((nl)/2);
    middlec=((nc)/2);
    % kernel filters expanded to size [nl,nc]
    B = zeros(nl,nc,L);
    % fft2 of kernels
    FBM = zeros(nl,nc,L);
    for i=1:L
        if d(i) > 1
            h = fspecial('gaussian',[dx,dy],sdf(i));
            B((middlel-dy/2+1:middlel+dy/2)-d(i)/2+1,(middlec-dx/2+1:middlec+dx/2)-d(i)/2+1,i) = h; %run
            % circularly center
            B(:,:,i)= fftshift(B(:,:,i));
            % normalize
            B(:,:,i) = B(:,:,i)/sum(sum(B(:,:,i)));
            FBM(:,:,i) = fft2(B(:,:,i));
        else
            B(1,1,i) = 1;
            FBM(:,:,i) = fft2(B(:,:,i));
        end
    end
end

function FBM2 = createConvKernelSubspace(sdf,nl,nc,L,dx,dy)

    %--------------------------------------------------------------------------
    %   Build convolution kernels FOR SUBSPACE!!!!
    %--------------------------------------------------------------------------
    %
    middlel=round((nl+1)/2);
    middlec=round((nc+1)/2);

    dx = dx+1;
    dy = dy+1;

    % kernel filters expanded to size [nl,nc]
    B = zeros(nl,nc,L);
    % fft2 of kernels
    FBM2 = zeros(nl,nc,L);

    s2 = max(sdf);
    for i=1:L
        if sdf(i) < s2
            h = fspecial('gaussian',[dx,dy],sqrt(s2^2-sdf(i)^2));
            B(middlel-(dy-1)/2:middlel+(dy-1)/2,middlec-(dx-1)/2:middlec+(dx-1)/2,i) = h;
    
            %circularly center
            B(:,:,i)= fftshift(B(:,:,i));
    
            % normalize
            B(:,:,i) = B(:,:,i)/sum(sum(B(:,:,i)));
            FBM2(:,:,i) = fft2(B(:,:,i));
        else
            % unit impulse
            B(1,1,i) = 1;
            FBM2(:,:,i) = fft2(B(:,:,i));
        end
    end
end

function X = ConvCM(X,FKM,nl,nc,L)

    if nargin == 3
        [L,n] = size(X);
        nc = n/nl;
    end
    X = conv2mat(real(ifft2(fft2(conv2im(X,nl,nc,L)).*FKM)));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % define a circular convolution (the same for all bands) accepting a
    % matrix  and returnig a matrix
    % size(X) is [no_bands_ms,n]
    % FKM is the  of the cube containing the fft2 of the convolution kernels
    % ConvCM = @(X,FKM)  reshape(real(ifft2(fft2(reshape(X', nl,nc,nb)).*FKM)), nl*nc,nb)';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function X = conv2mat(X,nl,nc,L)
    if ndims(X) == 3
        [nl,nc,L] = size(X);
        X = reshape(X,nl*nc,L)';
    elseif ndims(squeeze(X)) == 2
        L = 1;
        [nl,nc] = size(X);
        X = reshape(X,nl*nc,L)';
    end
end

function [M, Y] = createSubsampling(Yim,d,nl,nc,L)

    % subsampling matrix
    M = zeros(nl,nc,L);
    indexes = cell([L 1]);

    for i=1:L
        im = ones(floor(nl/d(i)),floor(nc/d(i)));
        maux = zeros(d(i));
        maux(1,1) = 1;
    
        M(:,:,i) = kron(im,maux);
        indexes{i} = find(M(:,:,i) == 1);
        Y(i,indexes{i}) = conv2mat(Yim{i},nl/d(i),nc/d(i),1);
    end
end

function [Yim] = unnormaliseData(Yim, av)
    if iscell(Yim)
        % mean squared power = 1
        nb = length(Yim);    
        for i=1:nb
            Yim{i,1} = sqrt(Yim{i}.^2*av(i,1));
        end
    else
        nb = size(Yim,3);
        for i=1:nb
            Yim(:,:,i) = sqrt(Yim(:,:,i).^2*av(i,1));
        end
    end
end



function W = computeWeights(Y,d,sigmas,nl)

    % As in eq. (14) and (15)
    % Compute weigts for each pixel based on HR bands
    hr_bands = d==1;
    hr_bands = find(hr_bands)';
    for i=hr_bands
    %     grad(:,:,i) = imgradient(conv2im(Y(i,:),nl),'prewitt').^2;
    %     Intermediate gives also good results compared to prewitt
        grad(:,:,i) = imgradient(conv2im(Y(i,:),nl),'intermediate').^2;
    end
    grad = sqrt(max(grad,[],3));
    grad = grad / quantile(grad(:),0.95);

    Wim = exp(-grad.^2/2/sigmas^2);
    Wim(Wim<0.5) = 0.5;

    W = conv2mat(Wim,nl);
end

function X = conv2im(X,nl,nc,L)

    if size(X,2)==1
        X = conv2mat(X,nl,nc,L);
    end
    if nargin == 2
        [L,n] = size(X);
        if n==1
            X = conv2mat(X,nl,nc,L);
        end
        nc = n/nl;
    end
    X = reshape(X',nl,nc,L);
end

function [J,gradJ,AtAg] = grad_cost_G(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W)
    X=F*Z;
    BX=ConvCM(X,FBM,nl);
    HtHBX=Mask.*BX;
    ZH=ConvCM(Z,FDHC,nl);
    Zv=ConvCM(Z,FDVC,nl);
    ZHW=ZH.*W;
    ZVW=Zv.*W;
    grad_pen=ConvCM(ZHW,FDH,nl)+ConvCM(ZVW,FDV,nl);
    AtAg = F'*ConvCM(HtHBX,conj(FBM),nl)+2*tau*(q*ones(1,nl*nc)).*grad_pen;
    gradJ=AtAg-UBTMTy;
    J = 1/2 * sum( sum( Z .* AtAg ) ) - sum( sum( Z.*UBTMTy ) );     
end

function [ Z ] = CG(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W)
    maxiter = 1000;
    tolgradnorm = 0.1;%1e-6;    
    [cost,grad] = grad_cost_G(Z,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W);
    gradnorm = norm(grad(:));
    iter = 0;
    res = -grad;
    while ( gradnorm > tolgradnorm & iter < maxiter ) 
        iter = iter + 1;
       % fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);      
        if( iter == 1 )
            desc_dir = res;
        else
            beta = ( res(:).' * res(:) ) / ( old_res(:).' * old_res(:) );
            desc_dir = res + beta * desc_dir;
        end
        [~, ~, AtAp] = grad_cost_G(desc_dir,F,Y,UBTMTy,FBM,Mask,nl,nc,r,tau,q,FDH,FDV,FDHC,FDVC,W);
        alpha = ( res(:).' * res(:) ) / ( desc_dir(:).' * AtAp(:) );
        Z1 = Z + alpha * desc_dir;
        old_res = res;
        res = res - alpha* AtAp;
        gradnorm = norm( res(:) );
        % Transfer iterate info
        Z = Z1;
    end
end