class BaseMetadataPropertyModel:
    param_names = [] # override in inherited class

    def __init__(self, **kwargs):
        for p in self.param_names:
            if p in kwargs.keys():
                self.__setattr__(p, kwargs[p])
            else:
                self.__setattr__(p, 0)

    def is_valid(self):
        return all(hasattr(self, name) for name in self.param_names)

    def get_values(self):
        return [self.__getattribute__(name) for name in self.param_names]

    def number_of_params(self):
        return len(self.param_names)

    def get_params(self):
        return self.param_names



class BlastocystScore(BaseMetadataPropertyModel):
    param_names = ['BS.3', 'BS.4', 'BS.5', 'BS.6', 'BS.7', 'BS.8', 'BS.9', 'BS.10', 'BS.11', 'BS.12', 'BS.13', 'BS.14', 'BS.15', 'BS.17']

    def __init__(self, **kwargs):
        super(BlastocystScore, self).__init__(**kwargs)

class BlastocystGrade(BaseMetadataPropertyModel):
    param_names = ['ICM.A', 'ICM.A.', 'ICM.B', 'ICM.B.', 'ICM.B..C', 'ICM.C', 'ICM.N', 'TE.A', 'TE.A.', 'TE.B', 'TE.B.', 'TE.B..C', 'TE.C', 'TE.CM', 'TE.N', 'Expansion.1', 'Expansion.1.2', 'Expansion.2', 'Expansion.2.3', 'Expansion.3', 'Expansion.4', 'Expansion.5', 'Expansion.6', 'Expansion.CM', 'Expansion.N']

    def __init__(self, **kwargs):
        super(BlastocystGrade, self).__init__(**kwargs)

class Morphokinetics(BaseMetadataPropertyModel):
    param_names = ['tPnF', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'tM', 'tSB']

    def __init__(self, **kwargs):
        super(Morphokinetics, self).__init__(**kwargs)


class Metadata:
    def __init__(self, age: float, blastocyst_score: BlastocystScore, blastocyst_grade: BlastocystGrade, morphokinetics: Morphokinetics):
        self.age = age
        self.blastocyst_score = blastocyst_score
        self.blastocyst_grade = blastocyst_grade
        self.morphokinetics = morphokinetics

    def get_flat_length(self):
        return (1 if self.age else 0) + \
            (self.blastocyst_score.number_of_params() if self.blastocyst_score and self.blastocyst_score.is_valid() else 0) + \
            (self.blastocyst_grade.number_of_params() if self.blastocyst_grade and self.blastocyst_grade.is_valid() else 0) + \
            (self.morphokinetics.number_of_params() if self.morphokinetics and self.morphokinetics.is_valid() else 0)

    def flat_data(self):
        return ([self.age] if self.age else []) + \
            (self.morphokinetics.get_values() if self.morphokinetics and self.morphokinetics.is_valid() else []) + \
            (self.blastocyst_score.get_values() if self.blastocyst_score and self.blastocyst_score.is_valid() else []) + \
            (self.blastocyst_grade.get_values() if self.blastocyst_grade and self.blastocyst_grade.is_valid() else [])
