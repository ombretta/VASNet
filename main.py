__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

import os
import sys
import torch
import numpy as np
from scipy import stats
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init

from tensorboardX import SummaryWriter

from config import  HParameters
from sys_utils import print_table, print_pkg_versions
from vsum_tools import generate_summary, evaluate_summary
from vasnet_model import VASNet, i3d_SelfAttention, i3d_afterMaxPool3d_SelfAttention


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file


class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out

        
    def load_datasets(self, datasets = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets
        
        # print(datasets)
        # if ".txt" in datasets[0]: 
        #     with open(datasets[0], "r") as f:
        #         datasets = f.read().split("\n")
        #         if "" in datasets: datasets.remove("")

        datasets_dict = {}
        for dataset in datasets:
            # print(dataset)
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            # dataset_name = base_filename.split('_')[2]
            # print("\tDataset name:", dataset_name)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)

        return n_folds


    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        
        if not hps.finetune:
            print("Model: VASNet")
            self.model = VASNet()
            self.model.eval()
            self.model.apply(weights_init)
        
        elif hps.finetune and hps.backbone == "I3D":
            print("Model: I3D + VASNet")
            self.model = i3d_SelfAttention()
            self.model.eval()
            self.model.apply(weights_init) # Check if this works
            
            rgb_pt_checkpoint = "../../kinetics_i3d_pytorch/model/model_rgb.pth"
            self.model.I3D.load_state_dict(torch.load(rgb_pt_checkpoint))
            
            # Don't finetune the first convolutional layers: it's too computational expensive

            params_not_to_train = ["conv3d_1a", "conv3d_2b", "conv3d_2c", "mixed_3b", "mixed_3c", "mixed_4", "mixed_5"]
            for name, param in self.model.I3D.named_parameters():
                    if any(param in name for param in params_not_to_train):
                        # print(name, param)
                        param.requires_grad = False
                        
        elif hps.finetune and hps.backbone == "I3D_afterMaxPool3d":
            print("Model: I3D_afterMaxPool3d + VASNet")
            self.model = i3d_afterMaxPool3d_SelfAttention()
            self.model.eval()
            self.model.apply(weights_init) # Check if this works
            
            rgb_pt_checkpoint = "../../kinetics_i3d_pytorch/model/model_rgb.pth"
            # self.model.I3D.load_state_dict(torch.load(rgb_pt_checkpoint))
            i3d_checkpoint = torch.load(rgb_pt_checkpoint)
            model_dict = self.model.I3D_after_maxPool3d.state_dict()
            model_dict.update({k.replace("module.",""):i3d_checkpoint[k] \
                              for k in i3d_checkpoint.keys() \
                              if ("mixed_4" in k or "mixed_5" in k)})
            self.model.I3D_after_maxPool3d.load_state_dict(model_dict)
            
        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file


    def train(self, output_dir='EX-0'):
        
        writer = SummaryWriter(log_dir="./"+output_dir+"/runs/lr"+str(self.hps.lr[0])+"_wd"+str(self.hps.l2_req))

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = torch.nn.MSELoss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req)

        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]

        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch)+"/"+str(self.hps.epochs_max)), end='')
            self.model.train()
            avg_loss = []

            random.shuffle(train_keys)

            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                
                # print(dataset)

                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores: target between 0 and 1
                target -= target.min()
                target /= target.max()
                
                if self.hps.use_cuda:
                    seq, target = seq.float().cuda(), target.float().cuda()

                seq_len = seq.shape[1]
                y, _ = self.model(seq,seq_len)
                
                # Added stochastic loss
                m = torch.distributions.bernoulli.Bernoulli(probs=0.5)
                theta = self.hps.coeff*m.sample()
                stochastic_y = y + theta
                
                loss_att = 0
                
                # Add regularization: y'
                loss = criterion(stochastic_y, target)
                # loss2 = y.sum()/seq_len
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss.detach_()

                avg_loss.append([float(loss.item()), float(loss_att)])
                
                del loss, y, seq, target

                torch.cuda.empty_cache() 

                # print(torch.cuda.memory_allocated())
            
            
            # Evaluate train dataset
            train_fscore, train_video_scores, _, _ = self.eval(self.train_keys)
            
            # If true, store generated summaries at every epoch (test results)
            if self.hps.store_intermediate_results:
                path, filename = os.path.split(self.split_file)
                base_filename, _ = os.path.splitext(filename)
                path = os.path.join(output_dir, 'temp_results', base_filename+'_'+str(self.split_id))
                os.makedirs(path, exist_ok=True)
                results_filename = os.path.join(path, str(epoch)+'.h5')
            else: results_filename = None
                
            # Evaluate test dataset
            val_fscore, video_scores, mean_kendall_corr_coeff, mean_spearman_corr_coeff = \
                self.eval(self.test_keys, 
                results_filename)
                
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch
                max_val_fscore_kcoeff = mean_kendall_corr_coeff
                max_val_fscore_scoeff = mean_spearman_corr_coeff

            avg_loss = np.array(avg_loss)
            print("   Train loss: {0:.05f}".format(np.mean(avg_loss[:, 0])), end='')
            print('   Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore), end='')
            print('   Test ranking coeffs Kendall/Spearman: {0:0.5}/{1:0.5}'.format(mean_kendall_corr_coeff, mean_spearman_corr_coeff))
            
            # Send losses and accuracies to tensorboard 
            curr_time = time.time()
            writer.add_scalar("loss/training", np.mean(avg_loss[:, 0]), epoch, curr_time)
            writer.add_scalar("fscore/training", train_fscore, epoch, curr_time)
            writer.add_scalar("fscore/validation", val_fscore, epoch, curr_time)
            writer.add_scalar("ranking_corr_coeff/validation/kendall", mean_kendall_corr_coeff, epoch, curr_time)
            writer.add_scalar("ranking_corr_coeff/validation/spearman", mean_spearman_corr_coeff, epoch, curr_time)

            if self.verbose:
                # video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[5, 35, 8, 8, 8])

            # Save model weights
            path, filename = os.path.split(self.split_file)
            base_filename, _ = os.path.splitext(filename)
            path = os.path.join(output_dir, 'models_temp', base_filename+'_'+str(self.split_id))
            os.makedirs(path, exist_ok=True)
            filename = str(epoch)+'_'+str(round(val_fscore*100,3))+'.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))
            
        writer.close()

        return max_val_fscore, max_val_fscore_epoch, max_val_fscore_kcoeff, max_val_fscore_scoeff


    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                data = self.get_data(key)
                # print(data)
                # seq = self.dataset[key]['features'][...]
                seq = data['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y, att_vec = self.model(seq, seq.shape[1])
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

        f_score, video_scores, mean_kendall_corr_coeff, mean_spearman_corr_coeff = \
                    self.eval_summary(summary, keys, metric=self.dataset_name,
                    results_filename=results_filename, 
                    att_vecs=att_vecs)

        return f_score, video_scores, mean_kendall_corr_coeff, mean_spearman_corr_coeff

    
    def ranking_corr_coeffs(self, machine_scores, gt_scores):

        # k_coeff = stats.kendalltau(machine_scores, gt_scores)
        # s_coeff = stats.spearmanr(machine_scores, gt_scores)
        
        sorted_machine_scores = np.argsort(machine_scores)
        sorted_gt_scores = np.argsort(gt_scores)
        
        k_coeff = stats.kendalltau(sorted_machine_scores, sorted_gt_scores)
        s_coeff = stats.spearmanr(sorted_machine_scores, sorted_gt_scores)

        return k_coeff[0], s_coeff[0]
    
    
    def eval_summary(self, machine_summary_activations, test_keys, 
                     results_filename=None, metric='tvsum', att_vecs=None):

        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms = []
        video_scores = []
        
        kendall_corr_coeffs = []
        spearman_corr_coeffs = []
        
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)
            
            k_coeff, s_coeff = self.ranking_corr_coeffs(probs, d['gtscore'][...])
            kendall_corr_coeffs.append(k_coeff)
            spearman_corr_coeffs.append(s_coeff)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm), 
                                 "{:.1%}".format(k_coeff), 
                                 "{:.1%}".format(s_coeff)])

            if results_filename:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)
                
                h5_res.create_dataset(key + '/kendall_coeff', data=k_coeff)
                h5_res.create_dataset(key + '/spearman_coeff', data=s_coeff)

                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)

                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)
        mean_kendall_corr_coeff = np.mean(kendall_corr_coeffs)
        mean_spearman_corr_coeff = np.mean(spearman_corr_coeffs)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores, mean_kendall_corr_coeff, mean_spearman_corr_coeff


#==============================================================================================



def eval_split(hps, splits_filename, output_file, data_dir='test'):

    print("\n")
    ao = AONet(hps)
    ao.initialize()
    ao.load_datasets()
    ao.load_split_file(splits_filename)

    val_fscores = []
    mean_kendall_corr_coeffs = []
    mean_spearman_corr_coeffs = []
    
    for split_id in range(len(ao.splits)):
        ao.select_split(split_id)
        weights_filename, _ = ao.lookup_weights_file(data_dir)
        print("Loading model:", weights_filename)
        ao.load_model(weights_filename)
        val_fscore, video_scores, mean_kendall_corr_coeff, mean_spearman_corr_coeff = \
            ao.eval(ao.test_keys, results_filename=data_dir+'/test_results.h5')
        val_fscores.append(val_fscore)
        mean_kendall_corr_coeffs.append(mean_kendall_corr_coeff)
        mean_spearman_corr_coeffs.append(mean_spearman_corr_coeff)

        val_fscore_avg = np.mean(val_fscores)
        mean_kendall_corr_coeffs_avg = np.mean(mean_kendall_corr_coeffs)
        mean_spearman_corr_coeffs_avg = np.mean(mean_spearman_corr_coeffs)
        
        if hps.verbose:
            video_scores = [["No.", "Video", "F-score", "K coeff", "S coeff"]] + video_scores
            print_table(video_scores, cell_width=[4,45,5])

        print("Avg F-score: ", val_fscore)
        print("Avg K coeff: ", mean_kendall_corr_coeff)
        print("Avg S coeff: ", mean_spearman_corr_coeff)
        print("")
        
        # Log F-score for this split_id
        output_file.write(splits_filename + ', ' + str(split_id) + ', ' + str(val_fscore) + '%\n')
        output_file.flush()
    
    print("Total AVG F-score: ", val_fscore_avg)
    print("Total AVG K coeff: ", mean_kendall_corr_coeffs_avg)
    print("Total AVG S coeff: ", mean_spearman_corr_coeffs_avg)
    output_file.write(splits_filename + "Total AVG F-score: " + str(val_fscore_avg) + '%\n')
    output_file.write(splits_filename + "Total AVG K coeff: " + str(mean_kendall_corr_coeffs_avg) + '%\n')
    output_file.write(splits_filename + "Total AVG S coeff: " + str(mean_spearman_corr_coeffs_avg) + '%\n')
    output_file.flush()

    return val_fscore_avg, mean_kendall_corr_coeffs_avg, mean_spearman_corr_coeffs_avg


def train(hps):
    os.makedirs(hps.output_dir, exist_ok=True)
    # os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    # os.makedirs(os.path.join(hps.output_dir, 'code'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    # os.system('cp -f splits/*.json  ' + hps.output_dir + '/splits/')
    # os.system('cp *.py ' + hps.output_dir + '/code/')

    # Create a file to collect results from all splits
    f = open(hps.output_dir + '/results.txt', 'wt')

    for split_filename in hps.splits:
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)
        print(dataset_name, dataset_type, splits)

        # For no augmentation use only a dataset corresponding to the split file
        datasets = None
        if dataset_type == '':
            datasets = hps.get_dataset_by_name(dataset_name)

        if datasets is None:
            datasets = hps.datasets

        f_avg = 0
        fscore_kcoeff_avg = 0
        fscore_scoeff_avg = 0
        
        n_folds = len(splits)
        for split_id in range(n_folds):
            ao = AONet(hps)
            ao.initialize()
            ao.load_datasets(datasets=datasets)
            ao.load_split_file(splits_file=split_filename)
            ao.select_split(split_id=split_id)

            fscore, fscore_epoch, fscore_kcoeff, fscore_scoeff = ao.train(output_dir=hps.output_dir)
            f_avg += fscore
            fscore_kcoeff_avg += fscore_kcoeff
            fscore_scoeff_avg += fscore_scoeff
            
            # Log F-score for this split_id
            f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(fscore_kcoeff) + ', ' + str(fscore_scoeff) + ', ' + str(fscore_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename)
            log_dir, _ = os.path.splitext(log_file)
            log_dir += '_' + str(split_id)
            log_file = os.path.join(hps.output_dir, 'models', log_dir) + '_' + str(fscore) + '.tar.pth'

            os.makedirs(os.path.join(hps.output_dir, 'models', ), exist_ok=True)
            os.system('mv ' + hps.output_dir + '/models_temp/' + log_dir + '/' + str(fscore_epoch) + '_*.pth.tar ' + log_file)
            os.system('rm -rf ' + hps.output_dir + '/models_temp/' + log_dir)

            print("Split: {0:}   Best F-score: {1:0.5f}   K coeff: {2:0.5f}   S coeff: {3:0.5f}   Model: {4:}".format(split_filename, fscore, fscore_kcoeff, fscore_scoeff, log_file))

        # Write average F-score for all splits to the results.txt file
        f_avg /= n_folds
        fscore_kcoeff_avg /= n_folds
        fscore_scoeff_avg /= n_folds
        
        f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg) + ', ' + str(fscore_kcoeff_avg) + ', ' + str(fscore_scoeff_avg) + '\n')
        f.flush()

    f.close()


if __name__ == "__main__":
    print_pkg_versions()

    parser = argparse.ArgumentParser("PyTorch implementation of paper \"Summarizing Videos with Attention\"")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
    parser.add_argument('-t', '--train', action='store_true', help="Train")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name") 
    parser.add_argument('--l2_req', type=float, default=0.00001, help="Weight_decay") 
    parser.add_argument('--lr_epochs', type=int, default=0, help="") 
    parser.add_argument('--lr', type=float, default=0.00005, help="Learning rate") 
    parser.add_argument('--epochs_max', type=int, default=300, help="Maximum number of epochs") 
    parser.add_argument('--coeff', type=float, default=0.0, help="Coefficient for Seyran's stochastic regularization term") 
    parser.add_argument('-f', '--finetune', action='store_true', help="Finetune i3d")    
    parser.add_argument('--backbone', type=str, help="Backbone used (I3D or I3D_afterMaxPool3d)")
    parser.add_argument('--store_intermediate_results', action='store_true', \
                        help="Wheter to store intermediate scores and machine summaries during training")    
    
    args = parser.parse_args()

    # MAIN
    #======================
    hps = HParameters()
    hps.load_from_args(args.__dict__)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)

    if hps.train:
        train(hps)
        
    results=[['No', 'Split', 'Mean F-score', 'K coeff', 'S coeff']]

    # Create a file to collect results from all splits
    output_file = open(hps.output_dir + '/test_results.txt', 'wt')

    for i, split_filename in enumerate(hps.splits):
        f_score, k_coeffs, s_coeffs = eval_split(hps, split_filename, output_file, data_dir=hps.output_dir)
        results.append([i+1, split_filename, str(round(f_score * 100.0, 3))+"%", 
        str(round(k_coeffs, 3)), str(round(s_coeffs, 3))])
    
    output_file.close()
    
    print("\nFinal Results:")
    print_table(results, cell_width=[3, 35, 8, 8, 8])
        
    sys.exit(0)

