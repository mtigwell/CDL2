import numpy as np

class MF():
    def __init__(self , rating_matrix ):
        self.num_u = rating_matrix.shape[0] #5551
        self.num_v = rating_matrix.shape[1] #16980
        self.u_lambda = 100
        self.v_lambda = 0.1
        self.k = 50 #latent維度
        self.a = 1
        self.b =0.01
        self.R = np.mat(rating_matrix)
        self.C = np.mat(np.ones(self.R.shape)) * self.b
        self.C[np.where(self.R>0)] = self.a
        self.I_U = np.mat(np.eye(self.k) * self.u_lambda)
        self.I_V = np.mat(np.eye(self.k) * self.v_lambda)
        self.U = np.mat(np.random.normal(0 , 1/self.u_lambda , size=(self.k,self.num_u)))
        self.V = np.mat(np.random.normal(0 , 1/self.v_lambda , size=(self.k,self.num_v)))
       

    def test(self):
        print( ((U_cut*self.R[np.ravel(np.where(self.R[:,j]>0)[1]),j] + self.v_lambda * self.V_sdae[j])).shape)
    
    def ALS(self , V_sdae):
        self.V_sdae = np.mat(V_sdae)
        
        V_sq = self.V * self.V.T * self.b
        for i in range(self.num_u):
            idx_a = np.ravel(np.where(self.R[i,:]>0)[1])
            V_cut = self.V[:,idx_a]
            self.U[:,i] = np.linalg.pinv( V_sq+ V_cut * V_cut.T * (self.a-self.b) + self.I_U )*(V_cut*self.R[i,idx_a].T) #V_sq+V_cut*V_cut.T*a_m_b = VCV^T
        
        U_sq = self.U * self.U.T * self.b
        for j in range(self.num_v):
            idx_a = np.ravel(np.where(self.R[:,j]>0)[1])
            U_cut = self.U[:,idx_a]
            self.V[:,j] = np.linalg.pinv(U_sq+U_cut*U_cut.T*(self.a-self.b)+self.I_V)* (U_cut*self.R[idx_a,j] + self.v_lambda * np.resize(self.V_sdae[j],(self.k,1)))
        
        return self.U ,self.V