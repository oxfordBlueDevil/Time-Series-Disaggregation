from sklearn.metrics import classification_report
import numpy as np

class ts_KnnClassifier(object):
    def __init__(self):
        '''
        preds is a list of predictions that will be made.
        plotter indicates whether to plot each nearest neighbor as it is found.
        '''
        self.preds=[]

    def predict(self, X_train, X_test, y_train, w):
        '''
        1-nearest neighbor classification algorithm using LB_Keogh lower 
        bound as similarity measure. Option to use DTW distance instead
        but is much slower.
        '''
        for i in X_test:
            min_dist = float('inf')
            closest_seq_ind = []

            for ind, j in enumerate(X_train):
                if self.LB_Keogh(i, j, 5) < min_dist:
                    dist = self.DTWDistance(i, j, w)
                    if dist < min_dist:
                        min_dist = dist
                        closest_seq_ind = j
            self.preds.append(y_train[closest_seq_ind])

    def performance(self, true_results):
        '''
        If the actual test set labels are known, can determine classification
        accuracy.
        '''
        return classification_report(true_results, self.preds)

    def get_preds(self):
        return self.preds

    def DTWDistance(self, s1, s2, w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW={}

        if w:
            w = max(w, abs(len(s1)-len(s2)))

            for i in range(-1,len(s1)):
                for j in range(-1,len(s2)):
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            if w:
                for j in range(max(0, i-w), min(len(s2), i+w)):
                    dist= (s1[i] - s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
            else:
                for j in range(len(s2)):
                    dist= (s1[i] - s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

        return np.sqrt(DTW[len(s1)-1, len(s2)-1])

    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind, i in enumerate(s1):

            lower_bound=min(s2[(ind-r if ind-r >= 0 else 0):(ind + r)])
            upper_bound=max(s2[(ind-r if ind-r >= 0 else 0):(ind + r)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound)**2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound)**2

        return np.sqrt(LB_sum)