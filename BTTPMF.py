import numpy as np
import pandas as pd
from functools import reduce

def get_matrix(userid_num,itemid_num,data):
    df = pd.DataFrame({
    'userId': userid_num,  
    'itemId': itemid_num, 
    'rating': data['Rating'].tolist()
})

    rating_matrix = df.pivot(index='itemId', columns='userId', values='rating')
    rating_matrix.fillna(0, inplace=True)
    rating_matrix_T = rating_matrix.T
    
    is_rating = rating_matrix.copy()
    is_rating[is_rating != 0] = 1
    return rating_matrix,rating_matrix_T,is_rating

class BTTPMF:
    def __init__(self, user_num, item_num, m, n, max_iter, rating_matrix, is_rating, epsilon, eta, lambad1,lambda2):
        self.user_num = user_num
        self.item_num = item_num
        self.m = m
        self.n = n
        self.max_iter = max_iter
        self.rating_matrix = rating_matrix
        self.is_rating = is_rating
        self.l1 = lambad1
        self.l2 = lambda2
        
        # Initialize epsilon and eta
        self.E = epsilon
        self.H = eta
        
        # Initialize user and item attributes
        self.user_persona = self.E.T
        self.item_nature = self.H.T
        
        # Initialize matrix A
        self.A = np.random.normal(loc=0, scale=1, size=(m, n))
        
    def fit(self):
        R = self.rating_matrix.values.ravel()
        I1, I2 = np.identity(self.m), np.identity(self.n)
            
        for iter_num in range(self.max_iter):
            for user_index in range(self.user_num):
                Fi = np.diag(self.is_rating[user_index])
                matrix_1 = [self.A, self.H, Fi, self.H.T, self.A.T]
                matrix_2 = [self.A, self.H, Fi, self.rating_matrix[user_index].T]
                ui = np.dot(np.linalg.inv(reduce(np.dot, matrix_1) + self.l1 * I1), (reduce(np.dot, matrix_2) +  self.l1 * self.user_persona[user_index].T))
                self.user_persona[user_index] = ui.T
            self.E = self.user_persona.T
            
            for item_index in range(self.item_num):
                Fj = np.diag(self.is_rating.iloc[item_index])
                matrix_3 = [self.A.T, self.E, Fj, self.E.T, self.A]
                matrix_4 = [self.A.T, self.E, Fj, self.rating_matrix.T[item_index].T]
                vj = np.dot(np.linalg.inv(reduce(np.dot, matrix_3) +  self.l2 * I2), (reduce(np.dot, matrix_4) + self.l2 * self.item_nature[item_index].T))
                self.item_nature[item_index] = vj.T
            self.H = self.item_nature.T
            
            gamma_matrix = np.zeros((self.m * self.n, self.user_num * self.item_num))
            for user_index in range(self.user_num):
                for p in range(self.m):
                    for item_index in range(self.item_num):
                        for q in range(self.n):
                            gamma_matrix[self.n * p + q][self.item_num * user_index + item_index] = self.E[p][user_index] * self.H[q][item_index]
            
            A1 = np.zeros((self.m, self.n))
            for p in range(self.m):
                for q in range(self.n):
                    A_pq = self.A[p][q]
                    for user_index in range(self.user_num):
                        for item_index in range(self.item_num):
                            A_pq += self.is_rating[user_index][item_index] * gamma_matrix[self.n * p + q][self.item_num * user_index + item_index] * (R[self.item_num * user_index + item_index] - np.dot(self.E.T[user_index], self.H.T[item_index]))
                    A1[p][q] = A_pq
            self.A[p][q] = A1[p][q]
        
        matri = [self.E.T, self.A, self.H]
        return matri
        
    def predict(self,matri):
        predict_score = reduce(np.dot, matri)
        MAE, count = 0, 0
        for i in range(self.user_num):
            for j in range(self.item_num):
                if self.rating_matrix[i][j] > 0:
                    wucha = abs(self.rating_matrix[i][j] - predict_score[i][j])
                    MAE += wucha
                    count += 1
        mae = MAE / count
        print(mae)
        return mae
