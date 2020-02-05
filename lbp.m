function LBP = lbp(I)
    [n, m] = size(I);
    LBP = zeros(n,m);
    bin = [1 2 4;8 0 16; 32 64 128]; %utile pour la conversion en décimal
    for i=2:n-1
        for j=2:m-1
            window = zeros(3);
            x=I(i,j);
            %Comparer chaque voisin au pixel selectionné
            window = I(i-1:i+1,j-1:j+1)-I(i,j);
            %Thresholder en fonction de cela
            window = window > 0;
            %Convertir en décimal
            window = window .* bin;
            %Calcul des 
            LBP(i,j) = sum(sum(window));
        end
    end
    LBP = histcounts(LBP, 256, 'normalization' , 'probability' );
end