import numpy as np
import random



class BTTM_Gibbs_Sampling():
    def __init__(self, K, P, Q, T, I, J, M, alpha, beta, xi, rho, user_data, item_data, text_data, max_iter):
        self.K = K 
        self.P = P 
        self.Q = Q 
        self.T = T 
        self.I = I 
        self.J = J 
        self.M = M 
        
        
        self.alpha = alpha
        self.beta = beta
        self.xi = xi
        self.rho = rho
        
        self.user_data = user_data
        self.item_data = item_data
        self.text_data = text_data
        
        self.max_iter = max_iter
        
        self.biterm_text = None
        self.persona = None
        self.nature = None
        self.topic = None
        
        self.user_persona = None
        self.user_persona_vector = None
        self.item_nature = None
        self.item_nature_vector = None
        self.persona_nature_topic = None
        self.persona_nature_topic_matrix = None
        self.topic_vocab = None
        self.topic_vocab_vector = None
        

        self.epsilon = None
        self.eta = None
        self.theta = None
        self.phi = None

    
    def get_biterm_text(self):
        biterm_text = []
        for text in self.text_data:
            biterm_list = []
            text_length = len(text)
            for i in range(text_length - 1):
                for j in range(i + 1, text_length):
                    if text[i] != text[j]:
                        biterm_list.append((text[i], text[j]))
            biterm_text.append(biterm_list)
        self.biterm_text = biterm_text
    
    def get_feature_matrix(self):
        persona_list, nature_list, topic_list = [], [], []
        for text in self.biterm_text:
            persona, nature, topic = [], [], []
            length = len(text)
            for index in range(length):
                random_persona = random.randint(0, self.P-1)
                random_nature = random.randint(0, self.Q-1)
                random_topic = random.randint(0, self.K-1)
                persona.append(random_persona)
                nature.append(random_nature)
                topic.append(random_topic)
            persona_list.append(persona)
            nature_list.append(nature)
            topic_list.append(topic)
        self.persona = persona_list
        self.nature = nature_list
        self.topic = topic_list

    def initial_matrix(self):
        #E
        self.user_persona = np.zeros((self.P, self.I))
        self.user_persona_vector = np.zeros(self.I)
        
        #H
        self.item_nature = np.zeros((self.Q, self.J))
        self.item_nature_vector = np.zeros(self.J)
        
        #A
        self.persona_nature_topic = np.zeros((self.K, self.P, self.Q))
        self.persona_nature_topic_matrix = np.zeros((self.P, self.Q))
        
    
        self.topic_vocab = np.zeros((self.T, self.K))
        self.topic_vocab_vector = np.zeros(self.K)
        
    
        for i in range(self.M):
            for j in range(len(self.biterm_text[i])): 
                v1 = self.biterm_text[i][j][0]
                v2 = self.biterm_text[i][j][1]
                x = self.persona[i][j]
                y = self.nature[i][j]
                z = self.topic[i][j]
                user_id = self.user_data[i]
                item_id = self.item_data[i]
                #increment user-persona count
                self.user_persona[x][user_id] += 1
                self.user_persona_vector[user_id] += 1
                #increment item-nature count
                self.item_nature[y][item_id] += 1
                self.item_nature_vector[item_id] += 1
                #increment (persona,nature)-topic count
                self.persona_nature_topic[z][x][y] += 1
                self.persona_nature_topic_matrix[x][y] += 1
                #increment topic-vocabulary1/2 count
                self.topic_vocab[v1][z] += 1
                self.topic_vocab[v2][z] += 1
                self.topic_vocab_vector[z] += 2

    def posterior_prob(self, persona, nature, topic, vocab1, vocab2, user, item):
        p_range = np.arange(self.P)
        q_range = np.arange(self.Q)
        k_range = np.arange(self.K)

        p_p = (self.persona_nature_topic[topic][p_range, nature]+self.alpha) / (self.persona_nature_topic_matrix[p_range, nature]+self.K*self.alpha) * (self.user_persona[p_range, user]+self.xi)
        p_q = (self.persona_nature_topic[topic][persona, q_range]+self.alpha) / (self.persona_nature_topic_matrix[persona, q_range]+self.K*self.alpha) * (self.item_nature[q_range, item]+self.rho)
        p_k = (self.topic_vocab[vocab1, k_range]+self.beta) * (self.topic_vocab[vocab2, k_range]+self.beta) / (self.topic_vocab_vector[k_range]+self.T*self.beta+1) / (self.topic_vocab_vector[k_range]+self.T*self.beta) * (self.persona_nature_topic[k_range, persona, nature]+self.alpha)

        return p_p.tolist(), p_q.tolist(), p_k.tolist()
    
    def search_posterior(self, prob_list):
        prob_array = np.array(prob_list)
        prob_array /= prob_array.sum()  # Normalize probabilities
        new_select = np.random.choice(len(prob_list), p=prob_array)
        return new_select
    

    def BTTM_Gibbs(self):
        Gibbs_result = []
        for iter_num in range(self.max_iter):
            for i in range(self.M):
                user_id = self.user_data[i]
                item_id = self.item_data[i]
                for j, (v1, v2) in enumerate(self.biterm_text[i]):
                    old_persona = self.persona[i][j]
                    old_nature = self.nature[i][j]
                    old_topic = self.topic[i][j]
                    
                    user_id = self.user_data[i]
                    item_id = self.item_data[i]
                    v1 = self.biterm_text[i][j][0]
                    v2 = self.biterm_text[i][j][1]
                    
                    self.user_persona[old_persona][user_id] -= 1
                    self.user_persona_vector[user_id] -= 1
                    self.item_nature[old_nature][item_id] -= 1
                    self.item_nature_vector[item_id] -= 1
                    self.persona_nature_topic[old_topic][old_persona][old_nature] -= 1
                    self.persona_nature_topic_matrix[old_persona][old_nature] -= 1
                    self.topic_vocab[v1][old_topic] -= 1
                    self.topic_vocab[v2][old_topic] -= 1
                    self.topic_vocab_vector[old_topic] -= 2
                    
                    posterior_prob_persona, posterior_prob_nature, posterior_prob_topic = self.posterior_prob(old_persona, old_nature, old_topic, v1, v2, user_id, item_id)
                    new_persona = self.search_posterior(posterior_prob_persona)
                    new_nature = self.search_posterior(posterior_prob_nature)
                    new_topic = self.search_posterior(posterior_prob_topic)

                    self.user_persona[new_persona][user_id] += 1
                    self.user_persona_vector[user_id] += 1
                    self.item_nature[new_nature][item_id] += 1
                    self.item_nature_vector[item_id] += 1
                    self.persona_nature_topic[new_topic][new_persona][new_nature] += 1
                    self.persona_nature_topic_matrix[new_persona][new_nature] += 1
                    self.topic_vocab[v1][new_topic] += 1
                    self.topic_vocab[v2][new_topic] += 1
                    self.topic_vocab_vector[new_topic] += 2
                    self.persona[i][j] = new_persona
                    self.nature[i][j] = new_nature
                    self.topic[i][j] = new_topic           
                
        Gibbs_result.append([self.user_persona, self.user_persona_vector, 
                    self.item_nature, self.item_nature_vector,
                    self.persona_nature_topic, self.persona_nature_topic_matrix,
                    self.topic_vocab, self.topic_vocab_vector])
        return Gibbs_result
    

  
    def calculate_parameter(self, Gibbs_result):
        self.epsilon = np.zeros((self.P, self.I))
        self.eta = np.zeros((self.Q, self.J))
        self.theta = np.zeros((self.K, self.P, self.Q))
        self.phi = np.zeros((self.T, self.K))
        for i in range(self.I):
            for p in range(self.P):
                self.epsilon[p][i] = (Gibbs_result[0][0][p][i]+self.xi)/(Gibbs_result[0][1][i]+self.P*self.xi)
        for j in range(self.J):
            for q in range(self.Q):
                self.eta[q][j] = (Gibbs_result[0][2][q][j]+self.rho)/(Gibbs_result[0][3][j]+self.Q*self.rho)
        for p in range(self.P):
            for q in range(self.Q):
                for k in range(self.K):
                    self.theta[k][p][q] = (Gibbs_result[0][4][k][p][q]+self.alpha)/(Gibbs_result[0][5][p][q]+self.K*self.alpha)
        for k in range(self.K):
            for t in range(self.T):
                self.phi[t][k] = (Gibbs_result[0][6][t][k]+self.beta)/(Gibbs_result[0][7][k]+self.T*self.beta)