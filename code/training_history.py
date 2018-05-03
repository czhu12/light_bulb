import numpy as np
from utils import ap

class TrainingHistory:
    def __init__(self):
        self.history = []
        self._should_save = False
        self._was_training = True

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
        # Early stopping strategy = stop training once validation loss doesn't
        # improve for n straight epochs. But in our case, we also need to consider
        # the amount of training data that currently exists.
        # We will use a rule based system:
        # If training data has increased by 20%.
        if len(self.history) <= 3:
            self._was_training = True
            return True

        recent_step = self.history[-1]
        greater_than_pct = current_labels / float(recent_step['num_labels']) > 1.2
        greater_than_val = current_labels - recent_step['num_labels'] > 30
        if greater_than_pct or greater_than_val:
            self._was_training = True
            return True

        # Continue training if test loss hasn't improved for 3 timesteps
        ts3 = self.history[-1]['test']['loss']
        ts2 = self.history[-2]['test']['loss']
        ts1 = self.history[-3]['test']['loss']
        should_keep_training = ts1 > ts2 and ts2 > ts3

        # Don't save model if we need to keep training
        just_stopped = self._was_training and not should_keep_training
        self._was_training = should_keep_training
        self._should_save = just_stopped

        return should_keep_training

    def should_reset_model(self, current_labels):
        if len(self.history) == 0:
            return False
        first_event = self.history[0]
        last_event = self.history[-1]
        return last_event['num_labels'] / first_event['num_labels'] > 5

    def should_save_model(self):
        ret = self._should_save
        self._should_save = False
        return ret

    def __len__(self):
        return len(self.history)

    def __getitem__(self, key):
        return self.history[key]

    def plot(self):
        test_loss = ap.markers['o']
        test_acc = ap.markers['s']
        p_loss = ap.AFigure(plot_labels=True)
        p_loss.plot(
            np.arange(len(self.history)),
            [step['test']['loss'] for step in self.history],
            marker='_o',
        )

        p_acc = ap.AFigure(plot_labels=True)
        p_acc.plot(
            np.arange(len(self.history)),
            [step['test']['acc'] for step in self.history],
            marker='_s',
        )

        return """
        Test Loss
        {}
        Test Accuracy
        {}
        """.format(p_loss.draw(), p_acc.draw())
