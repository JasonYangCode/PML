import random
import torch
from torchvision.transforms import functional as F

try:
    import accimage
except ImportError:
    accimage = None

__all__ = ["Compose", "ToPILImage", "ToTensor", "Roll", ]





class Compose:


    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToPILImage:

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):

        for i in range(pic.shape[0]):
            pic[i] = F.to_pil_image(pic[i], self.mode)

        return pic
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string

class ToTensor:


    def __call__(self, pic):
        return torch.from_numpy(pic).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Roll(torch.nn.Module):

    def __init__(self):
        super(Roll, self).__init__()
        self.off1 = random.randint(-5, 5)
        self.off2 = random.randint(-5, 5)

    def forward(self, img):
        data = torch.roll(img, shifts=(self.off1, self.off2), dims=(2, 3))
        return data
