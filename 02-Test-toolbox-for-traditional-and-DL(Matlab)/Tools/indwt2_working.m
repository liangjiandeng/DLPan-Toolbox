function X = indwt2_working(W,varargin)
%INDWT2 Inverse nondecimated 2-D wavelet transform.
%   INDWT2 will be removed in a future release of MATLAB. Use the
%   following function instead:
%       <a href="matlab:help iswt2">iswt2</a>

% Error in R2015a
% error(message('Wavelet:warnobsolete:ErrorReplaceINDWT2'));
nbIN = nargin-1;
idxCFS  = -1;
cfsFLAG = false;
if nbIN>0
    nbCELL = numel(W.dec);
    type = varargin{1};
    if ~ischar(type)
        error(message('Wavelet:FunctionArgVal:Invalid_ArgTyp'))
    end
    type = upper(type);
    cfsFLAG = isequal(upper(type(1)),'C');
    if cfsFLAG , type = type(2:end); end
    switch type
        case {'D','H'} ,           idxCFS = 0;
        case {'AA','LL','A','L'} , idxCFS = 1;
        case {'AD','LH'} ,         idxCFS = 2;
        case {'DA','HL'} ,         idxCFS = 3;
        case {'DD','HH'} ,         idxCFS = 4;
    end
    if nbIN>1 , levREC = varargin{2}; else levREC = W.level; end
        
    if idxCFS>1
        idxCFS = idxCFS + 3*(W.level-levREC);
        if ~cfsFLAG
            for j=1:nbCELL
                if ~isequal(j,idxCFS);
                    W.dec{j} = zeros(size(W.dec{j}));
                end
            end
        else
            X = W.dec{idxCFS};   % Coefficients
            return
        end
        
    elseif idxCFS==1   % Approximations (AA or LL)
        if cfsFLAG && levREC==W.level 
            X = W.dec{1}; 
            return; % Coefficients of Approximation at level MAX
        end
        idxMinToKill = 1 + 3*(W.level-levREC)+1;
        for j=idxMinToKill:nbCELL
            W.dec{j} = zeros(size(W.dec{j}));
        end
                
    elseif idxCFS==0
        idxMaxToKill = 1 + 3*(W.level-levREC);
        for j=1:idxMaxToKill
            W.dec{j} = zeros(size(W.dec{j}));
        end
        
    else
        
    end
end

% Initialization.
Lo  = W.filters.LoR;
Hi  = W.filters.HiR;
dwtEXTM = W.mode;
perFLAG = isequal(dwtEXTM,'per');
cfs   = W.dec;
sizes = W.sizes;
level = W.level;

maxloop = level;
if idxCFS==1 && cfsFLAG , maxloop = (level-levREC); end

idxBeg = 1;
for k=1:maxloop
    idxEnd = idxBeg+3;
    dec = reshape(cfs(idxBeg:idxEnd),2,2);
    sizerec = sizes(k+1,:);
    X   = recFUNC(dec,sizerec,Lo,Hi,perFLAG);
    cfs(1:idxEnd-1) = {[]};
    cfs{idxEnd} = X;
    idxBeg = idxEnd;
end

if abs(idxCFS)==1 && ~cfsFLAG && length(W.sizeINI)==3
    % X = uint8(X);
end
%-----------------------------------------------------------------------%
function X = recFUNC(dec,sINI,Lo,Hi,perFLAG)

% Reconstruction.
perm = [2,1,3];
W = cell(1,2);
for i = 1:2
    W{i} = wrec1D(dec{i,1},Lo{2},perm,perFLAG) + ...
        wrec1D(dec{i,2},Hi{2},perm,perFLAG);
end
X = (wrec1D(W{1},Lo{1},[],perFLAG) + wrec1D(W{2},Hi{1},[],perFLAG))/4;

% Extraction of central part
sREC = size(X);
F = floor((sREC-sINI)/2);
C = ceil((sREC-sINI)/2);
X = X(1+F(1):end-C(1),1+F(2):end-C(2),:);
%-----------------------------------------------------------------------%
function X = wrec1D(X,F,perm,perFLAG)

if ~isempty(perm) , X = permute(X,perm); end
if perFLAG
    nb = length(F)-1;
    X = [X X(:,1:nb,:)];
end
X = convn(X,F);
if ~isempty(perm) , X = permute(X,perm); end
%-----------------------------------------------------------------------%
