function varargout = ndwt2_working(X,level,varargin)
%NDWT2 Nondecimated 2-D wavelet transform.
%   NDWT2 will be removed in a future release of MATLAB. Use the
%   following function instead:
%       <a href="matlab:help swt2">swt2</a>

% Error in R2015a
% error(message('Wavelet:warnobsolete:ErrorReplaceNDWT2'));
nbIn = length(varargin);
if nbIn < 1
    error(message('MATLAB:narginchk:notEnoughInputs'));
elseif nbIn > 5
    error(message('MATLAB:narginchk:tooManyInputs'));
end

LoD = cell(1,2); HiD = cell(1,2); LoR = cell(1,2); HiR = cell(1,2);
if ischar(varargin{1})
    [LD,HD,LR,HR] = wfilters(varargin{1}); 
    for k = 1:2
        LoD{k} = LD; HiD{k} = HD; LoR{k} = LR; HiR{k} = HR;
    end

elseif isstruct(varargin{1})
    if isfield(varargin{1},'w1') && isfield(varargin{1},'w2')
        for k = 1:2
            [LoD{k},HiD{k},LoR{k},HiR{k}] = ...
                wfilters(varargin{1}.(['w' int2str(k)]));
        end
    elseif isfield(varargin{1},'LoD') && isfield(varargin{1},'HiD') && ...
           isfield(varargin{1},'LoR') && isfield(varargin{1},'HiR')
        for k = 1:2
            LoD{k} = varargin{1}.LoD{k}; HiD{k} = varargin{1}.HiD{k};
            LoR{k} = varargin{1}.LoR{k}; HiR{k} = varargin{1}.HiR{k};
        end
    else
        error(message('Wavelet:FunctionArgVal:Invalid_ArgVal'));
    end
        
elseif iscell(varargin{1})
    if ischar(varargin{1}{1})
        for k = 1:2
            [LoD{k},HiD{k},LoR{k},HiR{k}] = wfilters(varargin{1}{k});
        end
    else
        LoD(1:end) = varargin{1}(1); HiD(1:end) = varargin{1}(2);
        LoR(1:end) = varargin{1}(3); HiR(1:end) = varargin{1}(4);
    end
else
    
end
nextArg = 2;

dwtEXTM = 'sym';
while nbIn>=nextArg
    argName = varargin{nextArg};
    argVal  = varargin{nextArg+1};
    nextArg = nextArg + 2;
    switch argName
        case 'mode' , dwtEXTM = argVal;
    end
end

% Initialization.
if isempty(X) , varargout{1} = []; return; end
sX = size(X);
X = double(X);
sizes = zeros(level+1,length(sX));
sizes(level+1,:) = sX;

for k=1:level
    dec = decFUNC(X,LoD,HiD,dwtEXTM);
    X = dec{1,1,1};
    sizes(level+1-k,:) = size(X);
    dec = reshape(dec,4,1,1);
    if k>1
        cfs(1) = [];
        cfs = cat(1,dec,cfs);
    else
        cfs = dec;
    end
end

WT.sizeINI = sX;
WT.level = level;
WT.filters.LoD = LoD;
WT.filters.HiD = HiD;
WT.filters.LoR = LoR;
WT.filters.HiR = HiR;
WT.mode = dwtEXTM;
WT.dec = cfs;
WT.sizes = sizes;
varargout{1} = WT;

%-------------------------------------------------------------------------%
function dec = decFUNC(X,LoD,HiD,dwtEXTM)

dec = cell(2,2);
permVect = [];
[a_Lo,d_Hi] = wdec1D(X,LoD{1},HiD{1},permVect,dwtEXTM);
permVect = [2,1,3];
[dec{1,1},dec{1,2}] = wdec1D(a_Lo,LoD{2},HiD{2},permVect,dwtEXTM);
[dec{2,1},dec{2,2}] = wdec1D(d_Hi,LoD{2},HiD{2},permVect,dwtEXTM);
%-------------------------------------------------------------------------%
function [L,H] = wdec1D(X,Lo,Hi,perm,dwtEXTM)

if ~isempty(perm) , X = permute(X,perm); end
sX = size(X);
if length(sX)<3 , sX(3) = 1; end
lf = length(Lo);
lx = sX(2);
lc = lx+lf-1;
switch dwtEXTM
    case 'zpd'             % Zero extension.
        
    case {'sym','symh'}    % Symmetric extension (half-point).
        X = [X(:,lf-1:-1:1,:) , X , X(:,end:-1:end-lf+1,:)];
        
    case 'sp0'             % Smooth extension of order 0.
        X = [X(:,ones(1,lf-1),:) , X , X(:,lx*ones(1,lf-1),:)];
        
    case {'sp1','spd'}     % Smooth extension of order 1.
        Z = zeros(sX(1),sX(2)+ 2*lf-2,sX(3));
        Z(:,lf:lf+lx-1,:) = X;
        last = sX(2)+lf-1;
        for k = 1:lf-1
            Z(:,last+k,:) = 2*Z(:,last+k-1,:)- Z(:,last+k-2,:);
            Z(:,lf-k,:)   = 2*Z(:,lf-k+1,:)- Z(:,lf-k+2,:);
        end
        X = Z; clear Z;
        
    case 'symw'            % Symmetric extension (whole-point).
        X = [X(:,lf:-1:2,:) , X , X(:,end-1:-1:end-lf,:)];
        
    case {'asym','asymh'}  % Antisymmetric extension (half-point).
        X = [-X(:,lf-1:-1:1,:) , X , -X(:,end:-1:end-lf+1,:)];        
        
    case 'asymw'           % Antisymmetric extension (whole-point).
        X = [-X(:,lf:-1:2,:) , X , -X(:,end-1:-1:end-lf,:)];

    case 'rndu'            % Uniformly randomized extension.
        X = [randn(sX(1),lf-1,sX(3)) , X , randn(sX(1),lf-1,sX(3))];        
                        
    case 'rndn'            % Normally randomized extension.
        X = [randn(sX(1),lf-1,sX(3)) , X , randn(sX(1),lf-1,sX(3))];        
                
    case 'ppd'             % Periodized extension (1).
        X = [X(:,end-lf+2:end,:) , X , X(:,1:lf-1,:)];
        
    case 'per'             % Periodized extension (2).
        if rem(lx,2) , X = [X , X(:,end,:)]; end
        X = [X(:,end-lf+2:end,:) , X , X(:,1:lf-1,:)];        
end
L = convn(X,Lo);
H = convn(X,Hi);
clear X
switch dwtEXTM
    case 'zpd'
    otherwise
        lenL = size(L,2);
        first = lf; last = lenL-lf+1;
        L = L(:,first:last,:); H = H(:,first:last,:);
        lenL = size(L,2);
        first = 1+floor((lenL-lc)/2);  last = first+lc-1;
        L = L(:,first:last,:); H = H(:,first:last,:);
end
if isequal(dwtEXTM,'per')
    first = 1; last = lx;
    L = L(:,first:last,:);
    H = H(:,first:last,:);
end

if ~isempty(perm)
    L = permute(L,perm);
    H = permute(H,perm);
end
%-------------------------------------------------------------------------%


