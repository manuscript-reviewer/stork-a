from .metadata import Metadata

class InputImage:
    def __init__(self, filename: str, directory: str, label: float, metadata: Metadata):
        self.filename = filename
        self.directory = directory
        self.label = label
        self.metadata = metadata
