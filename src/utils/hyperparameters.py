import inspect


class HyperParameters:
    """The base class of hyperparameters."""

    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


if __name__ == "__main__":
    class B(HyperParameters):
        def __init__(self, a, b, c):
            self.save_hyperparameters(ignore=['c'])
            print('self.a =', self.a, 'self.b =', self.b)
            print('There is no self.c =', not hasattr(self, 'c'))

    b = B(a=1, b=2, c=3)
