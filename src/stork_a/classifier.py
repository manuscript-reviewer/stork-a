import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from .models.metadata import Metadata, BlastocystScore, BlastocystGrade, Morphokinetics

class Classifier:
    def __init__(self, name, model_name, base_path, path):
        self.__name = name
        self.__model_name = model_name
        self.__base_path = base_path
        self.__path = path

    @property
    def name(self):
        return self.__name

    @property
    def model_name(self):
        return self.__model_name

    @property
    def base_path(self):
        return self.__base_path

    @property
    def path(self):
        return self.__path

    def normalize_metadata(self, metadata: Metadata):
        train_data = pd.read_csv(os.path.join(self.base_path, 'train_data.txt'), delimiter = "\t")

        if not "Age" in self.name:
            metadata.age = None
        else:
            scaler = StandardScaler().fit([[value] for value in train_data['Age'].values])
            values = scaler.transform([[metadata.age]])
            metadata.age = values[0][0]

        if not "BlastocystScore" in self.name:
            metadata.blastocyst_score = None
        else:
            param_names = metadata.blastocyst_score.get_params()
            scaler = StandardScaler().fit(train_data[param_names].values)
            values = scaler.transform([metadata.blastocyst_score.get_values()])
            bs_data = { param_names[i]: values[0][i] for i in range(len(param_names)) }
            metadata.blastocyst_score = BlastocystScore(**bs_data)

        if not "BlastocystGrade" in self.name:
            metadata.blastocyst_grade = None
        else:
            param_names = metadata.blastocyst_grade.get_params()
            scaler = StandardScaler().fit(train_data[param_names].values)
            values = scaler.transform([metadata.blastocyst_grade.get_values()])
            bg_data = { param_names[i]: values[0][i] for i in range(len(param_names)) }
            metadata.blastocyst_grade = BlastocystGrade(**bg_data)

        if not "Morphokinetics" in self.name:
            metadata.morphokinetics = None
        else:
            param_names = metadata.morphokinetics.get_params()
            scaler = StandardScaler().fit(train_data[param_names].values)
            values = scaler.transform([metadata.morphokinetics.get_values()])
            m_data = { param_names[i]: values[0][i] for i in range(len(param_names)) }
            metadata.morphokinetics = Morphokinetics(**m_data)

