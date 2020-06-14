__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

from torch.autograd import Variable


class HParameters:

    def __init__(self):
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15

        self.l2_req = 0.00001
        self.lr_epochs = [0]
        self.lr = [0.00005]

        self.epochs_max = 300
        self.train_batch_size = 1

        self.output_dir = 'ex-10'

        self.root = ''
        self.datasets=['datasets/eccv16_dataset_summe_google_pool5.h5',
                       'datasets/eccv16_dataset_tvsum_google_pool5.h5',
                       'datasets/eccv16_dataset_ovp_google_pool5.h5',
                       'datasets/eccv16_dataset_youtube_google_pool5.h5',
                       '../SUM-GAN-AAE/data/SumMe/summe_i3d_mixed5c_aligned.h5',
                       '../SUM-GAN-AAE/data/TVSum/tvsum_i3d_mixed5c_aligned.h5']

        self.splits = ['splits/tvsum_splits.json',
                        'splits/summe_splits.json']

        self.splits += ['splits/tvsum_aug_splits.json',
                        'splits/summe_aug_splits.json']

        return


    def get_dataset_by_name(self, dataset_name):

        print(dataset_name)
        for d in self.datasets:
            
            print(d)
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()

                setattr(self, key, val)
            
            if key == "datasets" and ".txt" in args["datasets"]: 
                with open(args["datasets"], "r") as f:
                    self.datasets = f.read().split("\n")
                    if "" in self.datasets: self.datasets.remove("") #CHANGED HERE

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str


if __name__ == "__main__":

    # Tests
    hps = HParameters()
    print(hps)

    args = {'root': 'root_dir',
            'datasets': 'set1,set2,set3',
            'splits': 'split1, split2',
            'new_param_float': 1.23456
            }

    hps.load_from_args(args)
    print(hps)
