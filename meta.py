import os

from constants import FRAGMENTS_ALPHA, FRAGMENTS_BETA, ITERATION, LABEL_TYPE, ALPHA, BETA, IT_2_MODEL


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

    def get_current_label_dir(self):
        """The current label dir is always specified by the model prior to this iteration."""
        base_path = "./data/base_label_files/"
        label_type = self.get_current_label_type()
        model = self.get_previous_model()

        if model:
            return os.path.join(base_path, label_type, model)
        else:
            raise Exception(f"No prior model for iteration {self.iteration} ({self.get_current_phase()}) found")

    def get_current_model(self):
        return self.get_model_for_it(self.iteration)

    def get_previous_model(self):
        return self.get_model_for_it(self.iteration - 1)

    def get_current_fragments(self):
        phase = self.get_current_phase()
        if phase == ALPHA:
            return list(FRAGMENTS_ALPHA)
        elif phase == BETA:
            return list(FRAGMENTS_BETA)
        else:
            raise Exception("No such current phase:", phase)
