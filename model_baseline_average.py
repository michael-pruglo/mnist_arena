from arena_types import MNISTCompetitor
from fastai import *
from fastai.vision.all import *

class BaselineAverage(MNISTCompetitor):
    def __init__(self, loss_f):
        super().__init__()
        self.averages = []
        self.loss_f = loss_f

    def get_name(self):
        return "BaselineAverage"

    def train_on(self, training_set):
        self.averages = [digit_set.mean(axis=0) for digit_set in training_set]
        #show_image(torch.tensor(self.averages).permute((1,0,2)).reshape((28,10*28)), figsize=(14,15))

    def get_predictions(self, test_set):
        preds = []
        for digit_set in test_set:
            dig_preds = []
            for img in digit_set:
                losses = torch.tensor([self.loss_f(img, self.averages[d]) for d in range(10)])
                dig_preds.append(torch.argmin(losses))
            preds.append(dig_preds)
        return preds