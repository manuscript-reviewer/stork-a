import os

from .classifier import Classifier

current_file_dir = os.path.dirname(os.path.realpath(__file__))

class ClassifierFactory:
    def make(self, metadata_input, model_name):
        name = model_name # 'Abnormal-Normal', 'CxA-EUP', 'CxA-Everything'
        base_path = f"F__{model_name}"
        relative_path = "img_110_"
        if metadata_input['age']:
            relative_path += 'age_'
            name += ' Age'

            if metadata_input['morphokinetics']:
                relative_path += 'morpho_'
                name += ' Morphokinetics'

            if metadata_input['blastocyst_score']:
                relative_path += 'BS_'
                name += ' BlastocystScore'
            elif metadata_input['blastocyst_grade']:
                relative_path += 'QUAL_'
                name += ' BlastocystGrade'

        elif metadata_input['blastocyst_score']:
            relative_path += 'BS_'
            name += ' BlastocystScore'
        elif metadata_input['blastocyst_grade']:
            relative_path += 'QUAL_'
            name += ' BlastocystGrade'
        elif metadata_input['morphokinetics']:
            relative_path += 'morpho_'
            name += ' Morphokinetics'

        relative_path += model_name
        model_base_path = os.path.join(current_file_dir, base_path)
        classifier_path = os.path.join(current_file_dir, base_path, relative_path)
        return Classifier(name, model_name, model_base_path, classifier_path)
