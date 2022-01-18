class Result:
    def __init__(self, sample, good, confidence):
        self.sample = sample
        self.good = good
        self.confidence = confidence

    def get_string_result(self):
        if self.good:
            return 'Good'

        return 'Poor'
