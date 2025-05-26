class WandbSummarizer:
    def clear(self):
        self.log = {}

    def add(self, split, key, value):
        self.log[key] = value

    def add_dict(self, split, summary):
        for k, v in summary.items():
            self.log[f"{split}/{k}"] = v
