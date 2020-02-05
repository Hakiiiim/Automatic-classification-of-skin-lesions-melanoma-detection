function [accuracy,FScore] = metrics(Y_test,Y_test_asig)
    Cfman = zeros(2,2);

    for i = 1:length(Y_test)
       if (Y_test(i) == 1) && (Y_test_asig(i) == 1) 
           Cfman(1,1) = Cfman(1,1)+1;
       elseif (Y_test(i) == 0) && (Y_test_asig(i) == 1) 
           Cfman(2,1) = Cfman(2,1)+1;
       elseif (Y_test(i) == 1) && (Y_test_asig(i) == 0) 
           Cfman(1,2) = Cfman(1,2)+1;
       elseif (Y_test(i) == 0) && (Y_test_asig(i) == 0)
           Cfman(2,2) = Cfman(2,2)+1;
       end
    end

    Cf = Cfman;

    accuracy = trace(Cf)/sum(sum(Cf));

    Precision = Cf(1,1)/(Cf(1,1)+Cf(2,1));

    Recall = Cf(1,1)/(Cf(1,1)+Cf(1,2));

    FScore = 2*((Precision*Recall)/(Precision+Recall));
end

