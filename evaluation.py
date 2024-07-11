from gensim.models.coherencemodel import CoherenceModel

def BTTM_calculate_perplexity(theta, phi, docs, K):
    prob_list = []
    for doc_index in range(len(docs)):
        for vocab_index in docs[doc_index]:
            prob = 0
            for k in range(K):
                for persona in range(theta.shape[1]):
                    for nature in range(theta.shape[2]):
                        word_prob = theta[k][persona][nature]*phi[vocab_index][k]
                        prob += word_prob
                prob_list.append(prob)
    N = len(prob_list)
    BTTM_perplexity = 1
    for prob in prob_list:
        BTTM_perplexity *= 1 / pow(prob, 1 / N)
    return BTTM_perplexity

def compute_coherence_gensim(topics, texts, dictionary, coherence='c_v'):
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence)
    coherence_score = coherence_model.get_coherence()
    return coherence_score

def extract_topics_from_bttm(phi, dictionary, top_n=10):
    topics = []
    num_topics = phi.shape[1]
    for k in range(num_topics):
        top_words_idx = np.argsort(phi[:, k])[-top_n:]  # 获取高概率词项的索引
        top_words = [dictionary[i] for i in top_words_idx]
        topics.append(top_words)
    return topics