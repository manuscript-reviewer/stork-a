import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List

from .classifier import Classifier
from .my_model import MyModel
from .dataset_generator import DatasetGenerator
from .models.input_image import InputImage
from .models.result import Result
from .models.metadata import Metadata, BlastocystScore, BlastocystGrade, Morphokinetics

class StorkA:
    def __init__(self):
        self.data_transforms = {
            'test': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def eval(self, classifier: Classifier, input_images: List[InputImage]) -> List[Result]:
        # Create an image dataset
        image_datasets = DatasetGenerator(input_images, self.data_transforms['test'])

        # Create an loader to load image into GPU
        dataloader = DataLoader(image_datasets, shuffle=False, num_workers=4, drop_last=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        device = torch.device('cpu')
        model_ft = MyModel(input_images)
        model_ft.load_state_dict(torch.load(classifier.path + '/saved_model.pth', map_location=device))

        class_prob = []
        labs = []
        predict = []
        samples = []

        model_y = model_ft
        model_y.eval()

        with torch.no_grad():
            for inputs, labels_test, paths in dataloader:
                seq = labels_test[:, 1:]
                output = model_y(inputs.to(device), seq.to(device))
                _, predicted = torch.max(output.data, 1)
                output.cpu().numpy()
                prediction = output.softmax(dim=1)
                predict.append(predicted)
                class_prob.append(prediction)
                samples.extend(paths)

        confidence = torch.cat(class_prob)
        predicted = torch.cat(predict).type('torch.LongTensor')

        results = [Result(samples[i], bool(predicted[i]), confidence[i].tolist()) for i in range(len(samples))]
        return results
