from collections import Counter
import pandas as pd

################# Question 1 #################
def multinomial_nb(training_data, sms):# do not change the heading of the function
    class_dic = dict()
    for item in training_data:
        if item[1] not in class_dic:
            class_dic[item[1]] = 1
        else:
            class_dic[item[1]] += 1
    # the probility of "ham" ,"spam"occur in the whole document
    prob_ham ,prob_spam = class_dic['ham'] / sum(class_dic.values()),class_dic['spam'] / sum(class_dic.values())
    occur = dict()
    for key in training_data:
        if key[1] not in occur:
            occur[key[1]] = key[0]
        else:
            occur[key[1]] = Counter(occur[key[1]]) + Counter(key[0])

    V_num = len(set.union(set(occur['spam'].keys()), set(occur['ham'].keys())))

     # the likelihood of P(W|C)
    def pwc(count_w_c, count_c, V):
        return (count_w_c + 1) / (count_c + V)
    cond_prob_ham,cond_prob_spam = dict() , dict()
    for word in sms:
        if word not in occur['ham'] and word not in occur['spam']:
            continue
        category_ham = occur['ham']
        if word not in category_ham.keys():
            a = 0
        else:
            a = category_ham[word]
        b = sum(category_ham.values())
        pro_ham = pwc(a, b, V_num)
        cond_prob_ham[word] = pro_ham

        category_spam = occur['spam']
        if word not in category_spam.keys():
            x = 0
        else:
            x = category_spam[word]
        y = sum(category_spam.values())
        pro_spam = pwc(x, y, V_num)
        cond_prob_spam[word] = pro_spam

    def getFreq(sms):
        tokendic = {}
        for token in sms:
            if token not in tokendic:
                tokendic[token] = 1
            else:
                tokendic[token] += 1
        return tokendic
    wordFreq = getFreq(sms)
    p_ham ,p_spam = 1 , 1
    for key, val in wordFreq.items():
        try:
            p_ham *= cond_prob_ham[key] ** val
            p_spam *= cond_prob_spam[key] ** val
        except KeyError:
            pass
    ratio = (p_spam * prob_spam) / (p_ham * prob_ham)
    return ratio

    