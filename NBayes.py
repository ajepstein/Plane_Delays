import pandas as pd 
import numpy as np 
from scipy.stats import norm

class NBayes:

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.probs = dict()
        self.metrics = dict()
        self.vars = dict()

    """
    Function that generates a frequency table
    for a binary response for a given class.
    The frequencies of each are used as 
    conditional probabilies in the mixed
    Naive Bayes algorithm
    """
    def gen_frequency_table(self, x, y):
        x.loc[:, 'y'] = y
        s = pd.Series(x.iloc[:, 0])

        # Laplace smoothing with 2 outcomes, delayed or not 
        n = len(x) + self.alpha*2 
        x = x.set_index(x.columns.values[0])
        probs = { 
            1 : {},
            0 : {}
        }
        for c in s.unique():
            cs = x.loc[c]
            if str(type(cs)) == "<class 'pandas.core.series.Series'>":
                if cs['y'] == 0:
                    num_c_0 = 1; num_c_1 = 0
                else:
                    num_c_0 = 0; num_c_1 = 1
            else:    
                num_c_1 = len(cs.loc[(cs.y==1), :])
                num_c_0 = len(cs.loc[(cs.y==0), :])

            # Laplace smoothing
            num_c_0 += self.alpha; num_c_1 += self.alpha

            probs.get(1)[c] = num_c_1 / n
            probs.get(0)[c] = num_c_0 / n

        return probs

    def gen_gaussian_dist(self, x, y):
        x.loc[:, 'y'] = y
        probs = {
            1: dict(),
            0: dict()
        }
        
        ones = x.loc[(x.y==1), f'{x.columns.values[0]}'].tolist()
        zeros = x.loc[(x.y==0), f'{x.columns.values[0]}'].tolist()

        probs[1] = norm(np.mean(ones), np.std(ones) )
        probs[0] = norm(np.mean(zeros), np.std(zeros) )

        return probs

    def fit(self, df, y, multinomial=None, gaussian=None, binomial=None):
        if str(type(df)) != "<class 'pandas.core.frame.DataFrame'>":
            df = pd.DataFrame(df)
        # obtain prior response probabilities    
        ly = y.tolist()
        self.probs[1] = len([val for val in ly if val == 1]) / len(ly)
        self.probs[0] = len([val for val in ly if val == 0]) / len(ly)
        
        # store types of each var for predicting
        self.vars.update({
            'multinomial': multinomial,
            'gaussian': gaussian,
            'binomial': binomial
        })

        # fit all multinomial variables
        if multinomial is not None:
            for var in multinomial:
                x = df.loc[:, [var]]
                self.probs[var] = self.gen_frequency_table(x=x, y=y)
        # fit all Gaussian variables
        if gaussian is not None:
            for var in gaussian:
                x = df.loc[:, [var]]
                self.probs[var] = self.gen_gaussian_dist(x=x, y=y)
        # fit all binomial variables
        if binomial is not None:
            pass

    def predict(self, x_pred, y_pred=None):
        # obtain the log score of each row 
        # belonging to a certain class
        
        x_pred.loc[:, 'prob_0'] = x_pred.apply(lambda row: self.__predict_one_row__(row, 0), axis=1)
        x_pred.loc[:, 'prob_1'] = x_pred.apply(lambda row: self.__predict_one_row__(row, 1), axis=1)
        
        x_pred.loc[:, 'prediction'] = 0
        x_pred.loc[(x_pred.prob_0 < x_pred.prob_1), 'prediction'] = 1

        if y_pred is not None:
            eval_df = pd.DataFrame({
                'preds': x_pred.prediction,
                'truth': y_pred
            })
            eval_df_0 = eval_df.loc[(eval_df.preds == 0)]
            eval_df_1 = eval_df.loc[(eval_df.preds == 1)]

            FN = abs(sum(eval_df_0.preds - eval_df_0.truth))
            TN = len(eval_df_0) - FN

            FP = sum(eval_df_1.preds - eval_df_1.truth)
            TP = len(eval_df_1) - FP

            self.metrics.update({
                'precision': TP / (TP + FP),
                'recall': TP / (TP + FN),
                'specificity': TN / (TN + FP),
                'accuracy': (TN + TP) / (TN + FN + TN + TP),
            })
            self.metrics['F1'] = 2 * (
                 (self.metrics['precision']*self.metrics['recall']) / (self.metrics['precision'] + self.metrics['recall'])  
            )

        return x_pred.prediction

    def __predict_one_row__(self, row, outcome):
        # use sum of the logs of each
        # probability to avoid 
        # 'numerical underflow'
        prob = 0
        for col in self.vars['multinomial']: 
            col_prob = self.probs.get(col).get(outcome).get(row[col])
            # if an unseen category, use low probability
            if col_prob is None:
                # change this to as likely as 
                # most unprobable SEEN category
                col_prob = 0.01
            prob += np.log( col_prob )
        for col in self.vars['gaussian']:
            prob += self.probs.get(col).get(outcome).logpdf(row[col])
            
        # add the log of the prior
        prob += np.log(self.probs.get(outcome))
        return prob

    def get_metrics(self):
        for k, v in self.metrics.items(): print(k, ': ', v)