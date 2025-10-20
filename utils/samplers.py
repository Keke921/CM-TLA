import numpy as np
import random
from torch.utils.data.sampler import Sampler

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class RandomCycleIter:

    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler(Sampler):
    def __init__(self, data_source, num_samples_cls=4,):
        # pdb.set_trace()
        num_classes = data_source.num_classes
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        # self.num_samples = sum([len(x) for x in cls_data_list])
        self.num_samples_cls = num_samples_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
    

class DownSampler(Sampler):
    def __init__(self, data_source, n_max=100):
        self.num_classes = data_source.num_classes
        self.cls_data_list = [list() for _ in range(self.num_classes)]
        for i, label in enumerate(data_source.labels):
            self.cls_data_list[label].append(i)

        self.n_max = n_max
        self.cls_num_list = [min(n_max, len(x)) for x in self.cls_data_list]
        self.num_samples = sum(self.cls_num_list)
        
    def __iter__(self):
        data_list = []
        for y in range(self.num_classes):
            random.shuffle(self.cls_data_list[y])
            data_list.extend(self.cls_data_list[y][:self.n_max])
        random.shuffle(data_list)
        
        for i in range(self.num_samples):
            yield data_list[i]

    def __len__(self):
        return self.num_samples


class ReSampler(Sampler):
    def __init__(self, data_source, n_max=100):
        # pdb.set_trace()
        self.num_classes = data_source.num_classes

        cls_data_list = [list() for _ in range(self.num_classes)]
        for i, y in enumerate(data_source.labels):
            cls_data_list[y].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        cls_num_list = [len(x) for x in cls_data_list]

        self.sampled_cls_num_list = [min(n_max, n) for n in cls_num_list]
        cls_id_list = []
        for y in range(self.num_classes):
            cls_id_list.extend([y] * self.sampled_cls_num_list[y])
        self.cls_iter = RandomCycleIter(cls_id_list)

        self.num_samples = len(data_source.labels)
        
    def __iter__(self):
        for _ in range(self.num_samples):
            yield next(self.data_iter_list[next(self.cls_iter)])

    def __len__(self):
        return self.num_samples


class AugMixWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, augmix, transform):
        self.base_dataset = base_dataset
        self.augmix = augmix
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, label = self.base_dataset[index]

        # apply class-aware AugMix before transforms
        img = self.augmix(img, label)
        img = self.transform(img)
        return img, label


class ProbabilisticAugMix:
    def __init__(self, class_counts, severity_range=(1, 3), prob_range=(0.05, 0.55)):
        """
        class_counts: list[int]，每个类别样本数
        severity: AugMix 扰动强度（建议 3）
        prob_min, prob_max: 增强概率范围，和 class size 成反比
        """
        class_counts = np.array(class_counts, dtype=np.float32)
        inv_freq = 1.0 / class_counts
        inv_freq /= inv_freq.max()  # 归一化到 [0, 1]
        self.aug_probs = prob_range[0] + (prob_range[1] - prob_range[0]) * inv_freq
        aug_severity  = severity_range[0] + (severity_range[1] - severity_range[0]) * inv_freq
        self.aug_severity = np.round(aug_severity).astype(int)


        self.ops = [
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomAutocontrast(),
            T.RandomEqualize(),
            #T.RandomSolarize(threshold=192.0),
            T.RandomAdjustSharpness(0.5),
            T.RandomPosterize(bits=4),
            T.RandomRotation(10),
        ]

    def _apply_augmix(self, image: Image.Image, severity: int):
        ws = np.random.dirichlet([1] * 3)
        m = float(np.random.beta(1.0, 1.0))

        mix = Image.new("RGB", image.size)
        for i in range(3):
            image_aug = image.copy()
            ops_to_apply = random.sample(self.ops, k=severity)
            for op in ops_to_apply:
                image_aug = op(image_aug)
            mix = Image.blend(mix, image_aug, ws[i])
        return Image.blend(image, mix, m)

    def __call__(self, image: Image.Image, label: int):
        if random.random() < self.aug_probs[label]:
            return self._apply_augmix(image, self.aug_severity[label])
        else:
            return image

