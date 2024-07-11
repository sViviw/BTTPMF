from BTTM import BTTM_Gibbs_Sampling
from BTTPMF import BTTPMF, get_matrix
from dataset import process_reviews,get_dictionary
from evaluation import BTTM_calculate_perplexity, compute_coherence_gensim, extract_topics_from_bttm
import re

def main():
    # Step 1: Load and preprocess data,remember inter data path
    data, _ = process_reviews(data_path)
    uc = data['UserId'].value_counts() #344
    Ic = data['ItemId'].value_counts() #335

    userid = uc.index.tolist()
    itemid = Ic.index.tolist()

    user_dict = dict(zip(userid,range(len(userid))))
    item_dict = dict(zip(itemid,range(len(itemid))))

    userid_num = [user_dict[i] for i in data['UserId']]
    itemid_num = [item_dict[i] for i in data['ItemId']]

    words = data['Text'].tolist()
    words_list = []
    for text in words:
        text_words = re.findall(r'[a-zA-Z]+',text)
        words_list.append(text_words)
    words = [] #1479
    for text in words_list:
        for word in text:
            if word not in words:
                words.append(word)

    word_dict = dict(zip(words,range(len(words))))
    print(len(words))
    words_num = []
    for text in words_list:
        text_num = [word_dict[word] for word in text]
        words_num.append(text_num)
        
    dictionary,docs,texts = get_dictionary(data)

    #Step 2:train BTTM, example of number of topics
    bttm_coherence_scores = []
    bttm_perplexity_list = []
    bttpmf_mae_list = []
    topic_nums = [10,20,30,40,50,60,70]
    for num_topics in topic_nums:
        K = num_topics  
        P = 40    
        Q = 30    
        T = 1706 
        I = 344 
        J = 355 
        M = 350 
        alpha = 50/K
        beta = 0.01
        xi = 50/40
        rho = 50/30
        max_iter = 100

        bttm_model = BTTM_Gibbs_Sampling(K, P, Q, T, I, J, M, alpha, beta, xi, rho, user_data=userid_num, item_data = itemid_num, text_data =words_num, max_iter = max_iter)
        bttm_model.get_biterm_text()
        bttm_model.get_feature_matrix()
        bttm_model.initial_matrix()
        Gibbs_result = bttm_model.BTTM_Gibbs()
        bttm_model.calculate_parameter(Gibbs_result)
        bttm_topics = extract_topics_from_bttm(bttm_model.phi, dictionary)
        bttm_coherence = compute_coherence_gensim(bttm_topics, texts, dictionary, coherence='c_v')
        bttm_coherence_scores.append(bttm_coherence)
        bttm_perplexity = BTTM_calculate_perplexity(bttm_model.theta, bttm_model.phi, docs, num_topics)
        bttm_perplexity_list.append(bttm_perplexity)
     
        #Step 3:train BTTPMF, example of number of topics
        rating_matrix,_,is_rating = get_matrix(userid_num,itemid_num,data)
        bttpmf_model = BTTPMF(uc, Ic, P, Q, max_iter, rating_matrix, is_rating, bttm_model.epsilon, bttm_model.eta, lambad1 = 100, lambda2 = 100)
        EAH = bttpmf_model.fit()
        bttpmf_mae = bttpmf_model.predict(EAH)
        bttpmf_mae_list.append(bttpmf_mae)





if __name__ == '__main__':
    main()
