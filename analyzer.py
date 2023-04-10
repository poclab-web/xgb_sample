from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class XGBoostAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def importance(self):
        return None
    
    def calc_r2Score(self, y_true, y_pred, to_image=False):
        r2 = r2_score(y_true=y_true, y_pred=y_pred)

        # show predict properties to graph
        if to_image:
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(1)

            ax.scatter(y_true, y_pred)
            ax.set_xlabel('y_true')
            ax.set_ylabel('y_pred')

            fig.savefig('./r2_score.png')
            print(' r2 score plot saved!!')

        return r2