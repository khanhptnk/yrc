class WandbLogger:
    def clear(self):
        self.log = {}

    def __setitem__(self, key, value):
        self.log[key] = value

    def add(self, split, stats):
        for k, v in stats.items():
            self.log[f"{split}/{k}"] = v

    def get(self):
        return self.log
