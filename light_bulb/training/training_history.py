import numpy as np

class TrainingHistory:
    """
    This class keeps track of the state of representation learning,
    and epoch results.
    """
    def __init__(self):
        self.history = []
        self._should_save = False
        self._was_training = True
        self.finished_representation_learning = False

    @property
    def recent_num_samples(self):
        if len(self.history) == 0:
            return 0

        return self.history[-1]['stats']['labelled']['test']
    # TODO: Consider replacing num_labels idea with just creating a new model
    def add_train_eval_step(
        self,
        stats,
        train_acc,
        train_loss,
        test_acc,
        test_loss,
    ):
        self.history.append({
            'stats': stats,
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
        greater_than_pct = current_labels / float(recent_step['stats']['labelled']['train']) > 1.2
        greater_than_val = current_labels - recent_step['stats']['labelled']['train'] > 30
        if greater_than_pct or greater_than_val:
            self._was_training = True
            return True

        # Continue training if test loss hasn't improved for 3 timesteps
        ts3 = self.history[-1]['test']['loss']
        ts2 = self.history[-2]['test']['loss']
        ts1 = self.history[-3]['test']['loss']
        # Or if training out paces test for 3 timesteps
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
        return last_event['stats']['labelled']['train'] / first_event['stats']['labelled']['train'] > 5

    def should_save_model(self):
        ret = self._should_save
        self._should_save = False
        return ret

    def __len__(self):
        return len(self.history)

    def __getitem__(self, key):
        return self.history[key]
