
import torch
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights
import random
from pathlib import Path

class Resnet50FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(Resnet50FeatureExtractor, self).__init__()
        # loading the pre-trained weights and setting them to evaluation mode
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # writing hook function to capture activations at layers 2 and 3
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)
        # attaching the hooks to the final blocks of layer2 and layer3
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, input):

        self.features = [] # initialing the features store
        with torch.no_grad():
            _ = self.model(input)

        # processing the feature maps into a unified patch
        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        # aligning, concatenating, and flattening the feature maps into a column tensor
        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        resized_maps[0] = resized_maps[0] * 4
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        return patch
    
# ImageNet-standard normalization values for pre-trained models
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


def preprocessing_transform():
    return transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def load_patchcore_assets(path):
    # Load onto CPU specifically for Streamlit app efficiency
    assets = torch.load(path, map_location='cpu', weights_only=False)
    return assets




def load_sample_images(img_path):

    # 1. Define the directory path
    folder_path = Path(img_path)

    # 2. Grab all image files as a list of strings
    image_paths = [str(path) for path in folder_path.glob('*.png')]

    # 3. Select 10 random images at once
    # We use min() to handle cases where the folder has less than 10 images
    num_to_select = min(10, len(image_paths))
    random_10_images = random.sample(image_paths, k=num_to_select)
    return random_10_images

defect_classes = ["crack", "faulty_imprint", "good", "poke", "scratch", "squeeze"]

def get_defect_name(input_data):    
    """
    Extracts the word after the first underscore.
    Handles both strings and Streamlit UploadedFile objects.
    """
    try:
        # 1. Type Check: If it's a Streamlit UploadedFile, get the .name
        # Otherwise, assume it's already a string/path
        if hasattr(input_data, 'name'):
            filename = input_data.name
        else:
            filename = str(input_data)

        # 2. Logic to extract the fault
        defect_classes = ["crack", "faulty_imprint", "good", "poke", "scratch", "squeeze"]
        stem = Path(filename).stem
        parts = stem.split('_', 1)

        if len(parts) > 1:
            extracted_word = parts[1].lower()
            if extracted_word in defect_classes:
                return extracted_word
        
        return "unknown"

    except TypeError:
        # This catches the specific 'UploadedFile' vs 'PathLike' error
        return "unknown"
    except Exception:
        # Catch-all for any other weird string issues
        return "unknown"