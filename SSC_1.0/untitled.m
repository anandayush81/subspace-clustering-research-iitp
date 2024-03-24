
C = unique(A)
A= zeros(numel(C),row,col)
for m =1:numel(C)
    for n=1:row
        if data(n,col)==C(m)
            A(m,n,:)= data(n,:);
        end
    end
end
            
 c1= squeeze(A(1,:,:));
 c2= squeeze(A(2,:,:));
 c1( all(~c1,2), : ) = [];
 c2( ~all(c2,2), : ) = [];