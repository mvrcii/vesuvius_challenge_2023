import os

from utility.checkpoints import ITERATION, LABEL_TYPE, ALPHA, BETA, LABEL_BASE_PATH, LABEL_BINARIZED_PATH, \
    LABEL_BINARIZED_SINGLE_PATH, IT_2_MODEL
from utility.fragment import FRAGMENTS_BETA, FRAGMENTS_ALPHA


class AlphaBetaMeta:
    def __init__(self, iteration=None):
        self.iteration = ITERATION
        if iteration:
            self.iteration = iteration
        self.label_type = LABEL_TYPE

    def is_even(self):
        return self.iteration % 2 == 0

    def is_odd(self):
        return self.iteration % 2 != 0

    def get_current_phase(self):
        if self.is_odd():
            return ALPHA
        elif self.is_even():
            return BETA

    def get_current_label_type(self):
        return self.label_type

    @staticmethod
    def get_model_for_it(iteration):
        return IT_2_MODEL.get(iteration, None)

    def get_current_iteration(self):
        return self.iteration

    def get_current_base_label_dir(self):
        """The current label dir is always specified by the model prior to this iteration."""
        label_type = self.get_current_label_type()
        model = self.get_previous_model()

        if model:
            return os.path.join(LABEL_BASE_PATH, label_type, model)
        else:
            raise Exception(f"No prior model for iteration {self.iteration} ({self.get_current_phase()}) found")

    def get_current_binarized_label_dir(self, single=False):
        """The current label dir is always specified by the model prior to this iteration."""
        label_type = self.get_current_label_type()
        model = self.get_previous_model()

        if model:
            if single:
                return os.path.join(LABEL_BINARIZED_SINGLE_PATH, model)
            else:
                return os.path.join(LABEL_BINARIZED_PATH, label_type, model)
        else:
            # If there is no previous model, we are probably in te first iteration and thus we use the baseline labels
            return os.path.join(LABEL_BINARIZED_PATH, label_type, "baseline")

    def get_label_base_dir(self):
        return os.path.join(LABEL_BASE_PATH, self.get_current_label_type())

    def get_label_target_dir(self):
        return os.path.join(LABEL_BINARIZED_PATH, self.get_current_label_type())

    def get_current_model(self):
        return self.get_model_for_it(self.iteration)

    def get_previous_model(self):
        return self.get_model_for_it(self.iteration - 1)

    def get_current_train_fragments(self):
        phase = self.get_current_phase()

        assert not set(FRAGMENTS_ALPHA).intersection(set(FRAGMENTS_BETA)), "Fragments Alpha and Beta are overlapping!"

        if phase == ALPHA:
            return list(FRAGMENTS_ALPHA)
        elif phase == BETA:
            return list(FRAGMENTS_BETA)
        else:
            raise Exception("No such current phase:", phase)

    def get_current_inference_fragments(self):
        phase = self.get_current_phase()
        if phase == ALPHA:
            return list(FRAGMENTS_BETA)
        elif phase == BETA:
            return list(FRAGMENTS_ALPHA)
        else:
            raise Exception("No such current phase:", phase)
