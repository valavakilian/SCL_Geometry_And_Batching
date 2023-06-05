import torchvision
import torchvision.transforms as transforms
import numpy as np

# MNIST
class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True, transform=None,
                 target_transform=None, download=False, n_c_train_target = None, classes = None, n_maj = 5000):
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        self.num_per_cls_dict = dict()
        np.random.seed(rand_number)
        self.n_c_train_target = n_c_train_target
        self.classes = classes
        self.n_maj = n_maj
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = self.n_maj * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in self.classes:
                img_num_per_cls.append(self.n_c_train_target[cls_idx])
        else:
            img_num_per_cls.extend([int(self.n_maj)] * cls_num)
        print("img_num_per_cls: " + str(img_num_per_cls))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def extra_repr(self) -> str:
        return "Samples per class: {}".format(self.get_cls_num_list())
