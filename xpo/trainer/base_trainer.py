class BaseTrainer(object):
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    def step(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    def loss(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    def compute_rewards(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    def save_pretrained(self, save_directory):
        raise NotImplementedError("Not implemented")
