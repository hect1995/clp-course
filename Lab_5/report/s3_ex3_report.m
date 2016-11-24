%% A Posteriori Probability
%     P(B) = P_mail or P_spam
%     P(A) => A priori prob depending on DB
%     P(B|A) = S
%     P(A|B) = P(B|A)*P(A)/P(B)

    PBA = S
    
    if V_analisis_pred == 0 % Not SPAM
        PA = sum(Labs_train == 0)/length(Labs_train)
        PB = P_mail
    elseif V_analisis_pred == 1 % SPAM
        PA = sum(Labs_train == 1)/length(Labs_train)
        PB = P_spam
    end
        
    PAB = PBA*PA/PB