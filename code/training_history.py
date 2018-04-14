import numpy as np
from utils import ap

class TrainingHistory:
    def __init__(self):
        self.history = []
    
    # TODO: Consider replacing num_labels idea with just creating a new model
    def add_train_eval_step(
        self,
        num_labels,
        train_acc,
        train_loss,
        test_acc,
        test_loss,
    ):
        self.history.append({
            'num_labels': num_labels,
            'train': {
                'acc': train_acc,
                'loss': train_loss,
            },
            'test': {
                'acc': test_acc,
                'loss': test_loss,
            },
        })

    def should_continue_training(self, current_labels):
        # Early stopping strategy = stop training once validation loss doesn't improve for n straight epochs.
        # But in our case, we also need to consider the amount of training data that currently exists.
        # We will use a rule based system:
        # If training data has increased by 20%.
        if len(self.history) < 10:
            return True

        recent_step = self.history[-1]
        if current_labels / float(recent_step['num_labels']) > 1.1:
            return True

        # Continue training if train acc is still greater than test
        return recent_step['train']['acc'] > recent_step['test']['acc']

    def should_reset_model(self, current_labels):
        if len(self.history) == 0:
            return False
        first_event = self.history[0]
        last_event = self.history[-1]
        return last_event['num_samples'] / first_event['num_samples'] > 5

    def should_save_model(self):
        if len(self.history) == 0:
            return False

        return len(self.history) % 50 == 0

    def __len__(self):
        return len(self.history)

    def plot(self):
        p = ap.AFigure(plot_labels=True)
        p.plot(
            np.arange(len(self.history)),
            [step['train']['loss'] for step in self.history],
            marker='_o',
        )
        p.plot(
            np.arange(len(self.history)),
            [step['test']['loss'] for step in self.history],
            marker='_s',
        )
        return p.draw()
