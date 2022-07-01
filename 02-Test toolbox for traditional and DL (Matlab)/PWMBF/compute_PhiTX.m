function PhiTX=compute_PhiTX(X,L,h,type)
vec = @(x) x(:);
[M,T]=size(X);

switch lower(type)
    case 'dwt2'
        PhiTX=zeros(size(X));
        for k=1:T
            PhiTX(:,k)=vec(FWT2_PO(reshape(X(:,k),[sqrt(M) sqrt(M)]),log2(sqrt(M))-L,h));
        end
    case 'dwt'
        PhiTX=zeros(size(X));
        for k=1:T
            PhiTX(:,k)=vec(mdwt(reshape(X(:,k),[sqrt(M) sqrt(M)]),h,L));
        end
    case 'rwt'
        PhiTX=zeros((3*L+1)*M,T);
        for k=1:T
            [xl xh L]=mrdwt(reshape(X(:,k),[sqrt(M) sqrt(M)]),h,L);
            PhiTX(:,k)=vec([xl xh])/2;
        end
    case 'swt'
        PhiTX=zeros((3*L+1)*M,T);
        for k=1:T
            PhiTX(:,k)=vec(myswt2(reshape(X(:,k),[sqrt(M) sqrt(M)]),L,'db4'));
        end
    case 'iso'
        PhiTX=zeros((L+1)*M,T);
        for k=1:T
            PhiTX(:,k)=vec(cell2mat(atrousdec(reshape(X(:,k),[sqrt(M) sqrt(M)]),'maxflat',L)));
        end
    case 'cwt'
        J=L;
        [Faf, Fsf] = FSfarras; % 1st stage anal. & synth. filters
        [af, sf] = dualfilt1;
        for k=1:T
            w=dualtree2D(reshape(X(:,k),[sqrt(M) sqrt(M)]),J,Faf,af);
            W=[];
            for j=1:J
                W=[W' vec(w{j}{1}{1})']';
                W=[W' vec(w{j}{1}{2})']';
                W=[W' vec(w{j}{1}{3})']';
            end
            W=[W' vec(w{J+1}{1})']';
            for j=1:J
                W=[W' vec(w{j}{2}{1})']';
                W=[W' vec(w{j}{2}{2})']';
                W=[W' vec(w{j}{2}{3})']';
            end
            W=[W' vec(w{J+1}{2})']';
            PhiTX(:,k)=W;
        end
    case 'cplxdt'
        J=L;
        [Faf, Fsf] = FSfarras; % 1st stage anal. & synth. filters
        [af, sf] = dualfilt1;
        for k=1:T
            w=cplxdual2D(reshape(X(:,k),[sqrt(M) sqrt(M)]),J,Faf,af);
            W=[];
            for j=1:J
                W=[W' vec(w{j}{1}{1}{1})']';
                W=[W' vec(w{j}{1}{1}{2})']';
                W=[W' vec(w{j}{1}{1}{3})']';
                W=[W' vec(w{j}{1}{2}{1})']';
                W=[W' vec(w{j}{1}{2}{2})']';
                W=[W' vec(w{j}{1}{2}{3})']';
            end
            W=[W' vec(w{J+1}{1}{1})']';
            W=[W' vec(w{J+1}{1}{2})']';
            for j=1:J
                W=[W' vec(w{j}{2}{1}{1})']';
                W=[W' vec(w{j}{2}{1}{2})']';
                W=[W' vec(w{j}{2}{1}{3})']';
                W=[W' vec(w{j}{2}{2}{1})']';
                W=[W' vec(w{j}{2}{2}{2})']';
                W=[W' vec(w{j}{2}{2}{3})']';
            end
            W=[W' vec(w{J+1}{2}{1})']';
            W=[W' vec(w{J+1}{2}{2})']';
            PhiTX(:,k)=W;
        end
    otherwise
        error(['Unknown method ' type]);
end