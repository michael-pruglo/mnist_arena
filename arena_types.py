from abc import ABC, abstractmethod
 
class MNISTCompetitor(ABC):
    @abstractmethod
    def get_name(self): pass

    # training_set is a list of tensors shape (N x 28 x 28)
    @abstractmethod
    def train_on(self, training_set): pass

    # test_set is a list of tensors shape (N x 28 x 28)
    # return list of one-dimensional iterables containing N integers: predicted digits
    @abstractmethod
    def get_predictions(self, test_set): pass
