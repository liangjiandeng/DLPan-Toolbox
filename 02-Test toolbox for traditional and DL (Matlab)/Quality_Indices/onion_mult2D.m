%%%%%%%%%%%%%% Q2n aux. function
function ris = onion_mult2D(onion1,onion2)

[~,~,N3]=size(onion1);

if N3>1
   
    L=N3/2;

    a=onion1(:,:,1:L);
    b=onion1(:,:,L+1:end);
    b=cat(3,b(:,:,1),-b(:,:,2:end));
    c=onion2(:,:,1:L);
    d=onion2(:,:,L+1:end);
    d=cat(3,d(:,:,1),-d(:,:,2:end));


    if N3==2
        ris=cat(3,a.*c-d.*b,a.*d+c.*b); 
    else
        ris1=onion_mult2D(a,c);
        ris2=onion_mult2D(d,cat(3,b(:,:,1),-b(:,:,2:end)));
        ris3=onion_mult2D(cat(3,a(:,:,1),-a(:,:,2:end)),d);
        ris4=onion_mult2D(c,b);

        aux1=ris1-ris2;
        aux2=ris3+ris4;

        ris=cat(3,aux1,aux2);
    end
    
else
    ris = onion1.*onion2;   
end

end