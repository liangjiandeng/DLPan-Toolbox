%%%%%%%%%%%%%% Q2n aux. function
function ris=onion_mult(onion1,onion2)

N=length(onion1);

if N>1
  
    L=N/2;

    a=onion1(1:L);
    b=onion1(L+1:end);
    b=[b(1),-b(2:end)];
    c=onion2(1:L);
    d=onion2(L+1:end);
    d=[d(1),-d(2:end)];


    if N==2
        ris=[a*c-d*b,a*d+c*b];
    else
        ris1=onion_mult(a,c);
        ris2=onion_mult(d,[b(1),-b(2:end)]); %%
        ris3=onion_mult([a(1),-a(2:end)],d); %%
        ris4=onion_mult(c,b);

        aux1=ris1-ris2;
        aux2=ris3+ris4;

        ris=[aux1,aux2];
    end
   
else
    ris = onion1*onion2;
end

end