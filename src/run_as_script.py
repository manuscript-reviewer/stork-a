import os
import csv
import pandas as pd
from typing import List

from stork_a import StorkA, Classifier, Metadata, BlastocystScore, BlastocystGrade, Morphokinetics, InputImage, Result

def main():
    stork_a = StorkA()
    path = os.path.dirname(os.path.realpath(__file__))
    classifier_base_path = os.path.join(path, 'stork_a/F__Abnormal-Normal')
    classifier_path = os.path.join(path, 'stork_a/F__Abnormal-Normal/img_110_age_Abnorm-Norm') #select classifier type and which features to input by selecting the appropriate path

    classifier = Classifier('Abnormal-Normal', 'Abnormal-Normal', classifier_base_path, classifier_path)
    path_image_directory = os.path.join(path, 'all_110_imgs')
    path_dataset_file = os.path.join(classifier.path, 'model_input.txt')
    input_images = get_data_from_file(path_image_directory, path_dataset_file)
    results = stork_a.eval(classifier, input_images)

    predicted = pd.DataFrame([x.get_string_result() for x in results])
    confidence = pd.DataFrame([x.confidence for x in results])
    sample_file_names = pd.Series([os.path.basename(x.sample) for x in results])
    result = pd.concat([predicted, confidence, sample_file_names], axis=1, sort=False)
    result.columns = ['Predicted', 'Probability Aneuploid', 'Probability Euploid', 'Sample']
    print(result)
    # results_dir = classifier.path + '/Results'
    # os.mkdir(results_dir)
    # result.to_csv(results_dir + r'/EVALUATION.csv', index = False,  header=True)

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_data_from_file(path_image_directory: str, path_dataset_file: str) -> List[InputImage]:
    input_images = []
    with open(path_dataset_file, "r") as fs:
        reader = csv.DictReader(fs, delimiter='\t')

        for row in reader:
            params = { key: float(value) for key, value in row.items() if is_number(value) }
            metadata_input = {
                'age': params['EGG_AGE'] if 'EGG_AGE' in params else None,
                'blastocyst_score': BlastocystScore(**params) if all(elem in params for elem in BlastocystScore().get_params()) else None,
                'blastocyst_grade': BlastocystGrade(**params) if all(elem in params for elem in BlastocystGrade().get_params()) else None,
                'morphokinetics': Morphokinetics(**params) if all(elem in params for elem in Morphokinetics().get_params()) else None,
            }
            metadata = Metadata(**metadata_input)
            input_images.append(InputImage(row['img'], path_image_directory, float(row['Label']), metadata))

    return input_images

if __name__ == "__main__":
    main()
