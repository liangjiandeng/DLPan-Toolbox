function PhiX=compute_PhiX(X,L,h,type)
vec = @(x) x(:);
[N,r]=size(X);

switch lower(type)
    case 'dwt'
        M=N;
        PhiX=zeros(size(X));
        for k=1:r
            PhiX(:,k)=vec(midwt(reshape(X(:,k),[sqrt(M) sqrt(M)]),h,L));
        end
    case 'dwt2'
        M=N;
        PhiX=zeros(size(X));
        for k=1:r
            PhiX(:,k)=vec(IWT2_PO(reshape(X(:,k),[sqrt(M) sqrt(M)]),log2(sqrt(M))-L,h));
        end
    case 'rwt'
        M=N/(3*L+1);
        PhiX=zeros(M,r);
        for k=1:r
            PhiX(:,k)=vec(mirdwt(reshape(X(1:M,k),[sqrt(M) sqrt(M)]),reshape(X(M+1:end,k),[sqrt(M) 3*L*sqrt(M)]),h,L))*2;
        end
    case 'swt'
        M=N/(3*L+1);
        PhiX=zeros(M,r);
        for k=1:r
            PhiX(:,k)=vec(iswt2(reshape(X(:,k),[sqrt(M) sqrt(M) 3*L+1]),'db4'));
        end
    case 'iso'
        M=N/(L+1);
        PhiX=zeros(M,r);
        for k=1:r
            xc=reshape(X(:,k),[sqrt(M) (L+1)*sqrt(M)]);
            xc=mat2cell(xc,[sqrt(M)],repmat(sqrt(M),[1 L+1]));
            PhiX(:,k)=vec(atrousrec(xc,'maxflat'));
        end
    case 'cwt'
        J=L;
        [Faf, Fsf] = FSfarras; % 1st stage anal. & synth. filters
        [af, sf] = dualfilt1;
        
        PhiX=zeros(size(X,1)/2,size(X,2));
        n=sqrt(size(X,1)/2);
        for c=1:r
            
            W=X(:,c);
            j_offset=0;
            for j=1:J
                for k=1:3
                    w2{j}{1}{k}=reshape(W(j_offset+1+(k-1)*(n/2^j)^2:j_offset+k*(n/2^j)^2),[n/2^j n/2^j]);
                    w2{j}{2}{k}=reshape(W(j_offset+n^2+1+(k-1)*(n/2^j)^2:j_offset+n^2+k*(n/2^j)^2),[n/2^j n/2^j]);
                end
                j_offset=j_offset+3*(n/2^j)^2;
            end
            w2{J+1}{1}=reshape(W(n^2-(n/2^(J))^2+1:n^2),[n/2^J n/2^J]);
            w2{J+1}{2}=reshape(W(2*n^2-(n/2^(J))^2+1:2*n^2),[n/2^J n/2^J]);
            PhiX(:,c)=vec(idualtree2D(w2,J,Fsf,sf));
        end
    case 'cplxdt'
        J=L;
        [Faf, Fsf] = FSfarras; % 1st stage anal. & synth. filters
        [af, sf] = dualfilt1;
        
        PhiX=zeros(size(X,1)/4,size(X,2));
        n=sqrt(size(X,1)/4);
        for c=1:r
            
            W=X(:,c);
            j_offset=0;
            for j=1:J
                l_offset=0;
                for l=1:2
                    for k=1:3
                        w2{j}{1}{l}{k}=reshape(W(j_offset+l_offset+1+(k-1)*(n/2^j)^2:j_offset+l_offset+k*(n/2^j)^2),[n/2^j n/2^j]);
                        w2{j}{2}{l}{k}=reshape(W(j_offset+l_offset+2*n^2+1+(k-1)*(n/2^j)^2:j_offset+l_offset+2*n^2+k*(n/2^j)^2),[n/2^j n/2^j]);
                    end
                    l_offset=l_offset+3*(n/2^j)^2;
                end
                j_offset=j_offset+6*(n/2^j)^2;
            end
            w2{J+1}{1}{1}=reshape(W(2*n^2-2*(n/2^(J))^2+1:2*n^2-(n/2^(J))^2),[n/2^J n/2^J]);
            w2{J+1}{1}{2}=reshape(W(2*n^2-(n/2^(J))^2+1:2*n^2),[n/2^J n/2^J]);
            w2{J+1}{2}{1}=reshape(W(4*n^2-2*(n/2^(J))^2+1:4*n^2-(n/2^(J))^2),[n/2^J n/2^J]);
            w2{J+1}{2}{2}=reshape(W(4*n^2-(n/2^(J))^2+1:4*n^2),[n/2^J n/2^J]);
            PhiX(:,c)=vec(icplxdual2D(w2,J,Fsf,sf));
        end
    otherwise
        error(['Unknown method ' type]);
end

