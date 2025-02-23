from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from builtins import range
from collections import OrderedDict, defaultdict
import numpy as np
import random
import dill as pickle
from ADSeqGAN.data_loaders import Gen_Dataloader, Dis_Dataloader
from ADSeqGAN.generator import Generator
from ADSeqGAN.wgenerator import WGenerator
from ADSeqGAN.rollout import Rollout
from ADSeqGAN.discriminator import Discriminator
from ADSeqGAN.wdiscriminator import WDiscriminator
from ADSeqGAN.piror_classifier import prior_classifier

#from organ.discriminator import WDiscriminator as Discriminator

from tensorflow import logging
from rdkit import rdBase
import pandas as pd
from tqdm import tqdm, trange
import ADSeqGAN.mol_metrics
from collections import Counter


class ADSeqGAN(object):
    """Main class, where every interaction between the user
    and the backend is performed.
    """

    def __init__(self, name, metrics_module, params={},
                 verbose=True):
        """Parameter initialization.

        Arguments
        -----------

            - name. String which will be used to identify the
            model in any folders or files created.

            - metrics_module. String identifying the module containing
            the metrics.

            - params. Optional. Dictionary containing the parameters
            that the user whishes to specify.

            - verbose. Boolean specifying whether output must be
            produced in-line.

        """

        self.verbose = verbose

        # Set minimum verbosity for RDKit, Keras and TF backends
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.set_verbosity(logging.INFO)
        rdBase.DisableLog('rdApp.error')

        # Set configuration for GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # Set parameters
        self.PREFIX = name
        if 'WGAN' in params:
            self.WGAN = params['WGAN']
        else:
            self.WGAN = False

        if 'PRETRAIN_GEN_EPOCHS' in params:
            self.PRETRAIN_GEN_EPOCHS = params['PRETRAIN_GEN_EPOCHS']
        else:
            self.PRETRAIN_GEN_EPOCHS = 240

        if 'PRETRAIN_DIS_EPOCHS' in params:
            self.PRETRAIN_DIS_EPOCHS = params['PRETRAIN_DIS_EPOCHS']
        else:
            self.PRETRAIN_DIS_EPOCHS = 50

        if 'GEN_ITERATIONS' in params:
            self.GEN_ITERATIONS = params['GEN_ITERATIONS']
        else:
            self.GEN_ITERATIONS = 2

        if 'GEN_BATCH_SIZE' in params:
            self.GEN_BATCH_SIZE = params['GEN_BATCH_SIZE']
        else:
            self.GEN_BATCH_SIZE = 64

        if 'SEED' in params:
            self.SEED = params['SEED']
        else:
            self.SEED = None
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        if 'DIS_BATCH_SIZE' in params:
            self.DIS_BATCH_SIZE = params['DIS_BATCH_SIZE']
        else:
            self.DIS_BATCH_SIZE = 64

        if 'DIS_EPOCHS' in params:
            self.DIS_EPOCHS = params['DIS_EPOCHS']
        else:
            self.DIS_EPOCHS = 3

        if 'EPOCH_SAVES' in params:
            self.EPOCH_SAVES = params['EPOCH_SAVES']
        else:
            self.EPOCH_SAVES = 20

        if 'CHK_PATH' in params:
            self.CHK_PATH = params['CHK_PATH']
        else:
            self.CHK_PATH = os.path.join(
                os.getcwd(), 'checkpoints/{}'.format(self.PREFIX))

        if 'GEN_EMB_DIM' in params:
            self.GEN_EMB_DIM = params['GEN_EMB_DIM']
        else:
            self.GEN_EMB_DIM = 32

        if 'GEN_HIDDEN_DIM' in params:
            self.GEN_HIDDEN_DIM = params['GEN_HIDDEN_DIM']
        else:
            self.GEN_HIDDEN_DIM = 32

        if 'START_TOKEN' in params:
            self.START_TOKEN = params['START_TOKEN']
        else:
            self.START_TOKEN = 1

        if 'SAMPLE_NUM' in params:
            self.SAMPLE_NUM = params['SAMPLE_NUM']
        else:
            self.SAMPLE_NUM = 6400

        if 'CLASS_NUM' in params:
            self.CLASS_NUM = params['CLASS_NUM']
        else:
            self.CLASS_NUM = 1

        if 'BIG_SAMPLE_NUM' in params:
            self.BIG_SAMPLE_NUM = params['BIG_SAMPLE_NUM']
        else:
            self.BIG_SAMPLE_NUM = self.SAMPLE_NUM * 5

        if 'LAMBDA' in params:
            self.LAMBDA = params['LAMBDA']
        else:
            self.LAMBDA = 0.5
        if 'LAMBDA_C' in params:
            self.LAMBDA_C = params['LAMBDA_C']
        else:
            self.LAMBDA_C = 0.5

        # In case this parameter is not specified by the user,
        # it will be determined later, in the training set
        # loading.
        if 'MAX_LENGTH' in params:
            self.MAX_LENGTH = params['MAX_LENGTH']

        if 'DIS_EMB_DIM' in params:
            self.DIS_EMB_DIM = params['DIS_EMB_DIM']
        else:
            self.DIS_EMB_DIM = 64

        if 'DIS_FILTER_SIZES' in params:
            self.DIS_FILTER_SIZES = params['DIS_FILTER_SIZES']
        else:
            self.DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        if 'DIS_NUM_FILTERS' in params:
            self.DIS_NUM_FILTERS = params['DIS_NUM_FILTERS']
        else:
            self.DIS_NUM_FILTERS = [100, 200, 200, 200, 200, 100,
                                    100, 100, 100, 100, 160, 160]

        if 'DIS_DROPOUT' in params:
            self.DIS_DROPOUT = params['DIS_DROPOUT']
        else:
            self.DIS_DROPOUT = 0.75
        if 'DIS_GRAD_CLIP' in params:
            self.DIS_GRAD_CLIP = params['DIS_GRAD_CLIP']
        else:
            self.DIS_GRAD_CLIP = 1.0

        if 'WGAN_REG_LAMBDA' in params:
            self.WGAN_REG_LAMBDA = params['WGAN_REG_LAMBDA']
        else:
            self.WGAN_REG_LAMBDA = 1.0

        if 'DIS_L2REG' in params:
            self.DIS_L2REG = params['DIS_L2REG']
        else:
            self.DIS_L2REG = 0.2

        if 'TBOARD_LOG' in params:
            print('Tensorboard functionality')

        global mm
        if metrics_module == 'mol_metrics':
            mm = mol_metrics
        else:
            raise ValueError('Undefined metrics')

        self.AV_METRICS = mm.get_metrics()
        self.LOADINGS = mm.metrics_loading()

        self.PRETRAINED = False
        self.SESS_LOADED = False
        self.USERDEF_METRIC = False
        self.PRIOR_CLASSIFIER = False

    def load_training_set(self, file):
        """Specifies a training set for the model. It also finishes
        the model set up, as some of the internal parameters require
        knowledge of the vocabulary.

        Arguments
        -----------

            - file. String pointing to the dataset file.

        """

        # Load training set
        self.train_samples = mm.load_train_data(file)
        self.molecules, _ = zip(*self.train_samples)

        # Process and create vocabulary
        self.char_dict, self.ord_dict = mm.build_vocab(self.molecules, class_num=self.CLASS_NUM)
        
        # Calculate the quartiles of the length of the molecules
        self.len_list = [len(mol) for mol in self.molecules]
        self.average_len = np.mean(self.len_list)
        self.Q1 = np.percentile(self.len_list, 25)
        self.Q3 = np.percentile(self.len_list, 75)
        
        self.NUM_EMB = len(self.char_dict)
        self.PAD_CHAR = '_'
        self.PAD_NUM = self.char_dict[self.PAD_CHAR]
        self.DATA_LENGTH = max(map(len, self.molecules))
        print('Vocabulary:')
        print(list(self.char_dict.keys()))
        # If MAX_LENGTH has not been specified by the user, it
        # will be set as 1.5 times the maximum length in the
        # trining set.
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.molecules, key=len)) * 1.5)

        # Encode samples
        to_use = [sample for sample in self.train_samples
                  if mm.verified_and_below(sample[0], self.MAX_LENGTH)]
        molecules_to_use, label_to_use = zip(*to_use)
        positive_molecules = [mm.encode(sam,
                            self.MAX_LENGTH,
                            self.char_dict) for sam in molecules_to_use]
        self.positive_samples = [list(item) for item in zip(positive_molecules, label_to_use)]
        self.POSITIVE_NUM = len(self.positive_samples)
        self.TYPE_NUM = Counter([sam[1] for sam in to_use])

        # Print information
        if self.verbose:

            print('\nPARAMETERS INFORMATION')
            print('============================\n')
            print('Model name               :   {}'.format(self.PREFIX))
            print('Training set size        :   {} points'.format(
                len(self.train_samples)))
            print('Max data length          :   {}'.format(self.MAX_LENGTH))
            lens = [len(s[0]) for s in to_use]
            print('Avg Length to use is     :   {:2.2f} ({:2.2f}) [{:d},{:d}]'.format(
                np.mean(lens), np.std(lens), np.min(lens), np.max(lens)))
            print('Num valid data points is :   {}'.format(
                self.POSITIVE_NUM))
            print('Num different samples is :   {}'.format(
                self.TYPE_NUM))
            print('Size of alphabet is      :   {}'.format(self.NUM_EMB))
            print('')

            params = ['PRETRAIN_GEN_EPOCHS', 'PRETRAIN_DIS_EPOCHS',
                      'GEN_ITERATIONS', 'GEN_BATCH_SIZE', 'SEED',
                      'DIS_BATCH_SIZE', 'DIS_EPOCHS', 'EPOCH_SAVES',
                      'CHK_PATH', 'GEN_EMB_DIM', 'GEN_HIDDEN_DIM',
                      'START_TOKEN', 'SAMPLE_NUM', 'BIG_SAMPLE_NUM',
                      'LAMBDA', 'MAX_LENGTH', 'DIS_EMB_DIM',
                      'DIS_FILTER_SIZES', 'DIS_NUM_FILTERS',
                      'DIS_DROPOUT', 'DIS_L2REG']

            for param in params:
                string = param + ' ' * (25 - len(param))
                value = getattr(self, param)
                print('{}:   {}'.format(string, value))

        # Set model
        self.gen_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.dis_loader = Dis_Dataloader()
        self.mle_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        if self.WGAN:
            self.generator = WGenerator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                                        self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                                        self.MAX_LENGTH, self.START_TOKEN)
            self.discriminator = WDiscriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG,
                wgan_reg_lambda=self.WGAN_REG_LAMBDA,
                grad_clip=self.DIS_GRAD_CLIP)
        else:
            self.generator = Generator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                                       self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                                       self.MAX_LENGTH, self.START_TOKEN)
            self.discriminator = Discriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG,
                grad_clip=self.DIS_GRAD_CLIP)

        # run tensorflow
        self.sess = tf.InteractiveSession()
        #self.sess = tf.Session(config=self.config)

        #self.tb_write = tf.summary.FileWriter(self.log_dir)

    def define_metric(self, name, metric, load_metric=lambda *args: None,
                      pre_batch=False, pre_metric=lambda *args: None):
        """Sets up a new metric and generates a .pkl file in
        the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metric. Function taking as argument a sequence
            and returning a float value.

            - load_metric. Optional. Preprocessing needed
            at the beginning of the code.

            - pre_batch. Optional. Boolean specifying whether
            there is any preprocessing when the metric is applied
            to a batch of sequences. False by default.

            - pre_metric. Optional. Preprocessing operations
            for the metric. Will be ignored if pre_batch is False.

        Note
        -----------

            For combinations of already existing metrics, check
            the define_metric_as_combination method.

        """

        if pre_batch:
            def batch_metric(smiles, train_smiles=None):
                psmiles = pre_metric()
                vals = [mm.apply_to_valid(s, metric) for s in psmiles]
                return vals
        else:
            def batch_metric(smiles, train_smiles=None):
                vals = [mm.apply_to_valid(s, metric) for s in smiles]
                return vals

        self.AV_METRICS[name] = batch_metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [batch_metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def define_metric_as_combination(self, name, metrics, ponderations):
        """Sets up a metric made from a combination of
        previously existing metrics. Also generates a
        metric .pkl file in the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metrics. List containing the name identifiers
            of every metric in the list

            - ponderations. List of ponderation coefficients
            for every metric in the previous list.

        """

        funs = [self.AV_METRICS[metric] for metric in metrics]
        funs_load = [self.LOADINGS[metric] for metric in metrics]

        def metric(smiles, train_smiles=None, **kwargs):
            vals = np.zeros(len(smiles))
            for fun, c in zip(funs, ponderations):
                vals += c * np.asarray(fun(smiles))
            return vals

        def load_metric():
            return [fun() for fun in funs_load if fun() is not None]

        self.AV_METRICS[name] = metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        nmetric = [metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(nmetric, f)

    def define_metric_as_remap(self, name, metric, remapping):
        """Sets up a metric made from a remapping of a
        previously existing metric. Also generates a .pkl
        metric file in the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metric. String identifying the previous metric.

            - remapping. Remap function.

        """

        pmetric = self.AV_METRICS[metric]

        def nmetric(smiles, train_smiles=None, **kwargs):
            vals = pmetric(smiles, train_smiles, **kwargs)
            return remapping(vals)

        self.AV_METRICS[name] = nmetric
        self.LOADINGS[name] = self.LOADINGS[metric]

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [nmetric, self.LOADINGS[metric]]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def load_prev_user_metric(self, name, file=None):
        """Loads a metric that the user has previously designed.

        Arguments.
        -----------

            - name. String used to identify the metric.

            - file. String pointing to the .pkl file. Will use
            ../data/name.pkl by default.

        """
        if file is None:
            file = '../data/{}.pkl'.format(name)
        pkl = open(file, 'rb')
        data = pickle.load(pkl)
        self.AV_METRICS[name] = data[0]
        self.LOADINGS[name] = data[1]
        if self.verbose:
            print('Loaded metric {}'.format(name))

    def set_training_program(self, metrics=None, steps=None):
        """Sets a program of metrics and epochs
        for training the model and generating molecules.

        Arguments
        -----------

            - metrics. List of metrics. Each element represents
            the metric used with a particular set of epochs. Its
            length must coincide with the steps list.

            - steps. List of epoch sets. Each element represents
            the number of epochs for which a given metric will
            be used. Its length must coincide with the steps list.

        Note
        -----------

            The program will crash if both lists have different
            lengths.

        """

        # Raise error if the lengths do not match
        if len(metrics) != len(steps):
            return ValueError('Unmatching lengths in training program.')

        # Set important parameters
        self.TOTAL_BATCH = np.sum(np.asarray(steps))
        self.METRICS = metrics

        # Build the 'educative program'
        self.EDUCATION = {}
        i = 0
        for j, stage in enumerate(steps):
            for _ in range(stage):
                self.EDUCATION[i] = metrics[j]
                i += 1

    def load_metrics(self):
        """Loads the metrics."""

        # Get the list of used metrics
        met = list(set(self.METRICS))

        # Execute the metrics loading
        self.kwargs = {}
        for m in met:
            load_fun = self.LOADINGS[m]
            args = load_fun()
            if args is not None:
                if isinstance(args, tuple):
                    self.kwargs[m] = {args[0]: args[1]}
                elif isinstance(args, list):
                    fun_args = {}
                    for arg in args:
                        fun_args[arg[0]] = arg[1]
                    self.kwargs[m] = fun_args
            else:
                self.kwargs[m] = None

    def load_prev_pretraining(self, ckpt=None):
        """
        Loads a previous pretraining.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name_pretrain/pretrain_ckpt' is assumed.

        Note
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files, like in the following ls:

                checkpoint
                pretrain_ckpt.data-00000-of-00001
                pretrain_ckpt.index
                pretrain_ckpt.meta

            In this case, ckpt = 'pretrain_ckpt'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # Generate TF saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        # Load from checkpoint
        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt))
            self.PRETRAINED = True
        else:
            print('\t* No pre-training data found as {:s}.'.format(ckpt))

    def load_prev_training(self, ckpt=None):
        """
        Loads a previous trained model.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name/pretrain_ckpt' is assumed.

        Note 1
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files. An example ls:

                checkpoint
                validity_model_0.ckpt.data-00000-of-00001
                validity_model_0.ckpt.index
                validity_model_0.ckpt.meta
                validity_model_100.ckpt.data-00000-of-00001
                validity_model_100.ckpt.index
                validity_model_100.ckpt.meta
                validity_model_120.ckpt.data-00000-of-00001
                validity_model_120.ckpt.index
                validity_model_120.ckpt.meta
                validity_model_140.ckpt.data-00000-of-00001
                validity_model_140.ckpt.index
                validity_model_140.ckpt.meta

                    ...

                validity_model_final.ckpt.data-00000-of-00001
                validity_model_final.ckpt.index
                validity_model_final.ckpt.meta

            Possible ckpt values are 'validity_model_0', 'validity_model_140'
            or 'validity_model_final'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # If there is no Rollout, add it
        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.3, self.PAD_NUM)

        # Generate TF Saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Training loaded from previous checkpoint {}'.format(ckpt))
            self.SESS_LOADED = True
        else:
            print('\t* No training checkpoint found as {:s}.'.format(ckpt))

    def pretrain(self):
        """Pretrains generator and discriminator."""

        self.gen_loader.create_batches(self.positive_samples)
        # print("self.positive_samples:", self.positive_samples)
        # results = OrderedDict({'exp_name': self.PREFIX})

        if self.verbose:
            print('\nPRETRAINING')
            print('============================\n')
            print('GENERATOR PRETRAINING')

        t_bar = trange(self.PRETRAIN_GEN_EPOCHS)
        for epoch in t_bar:
            supervised_g_losses = []
            self.gen_loader.reset_pointer()
            for it in range(self.gen_loader.num_batch):
                batch = self.gen_loader.next_batch()
                x, class_label = zip(*batch)
                _, g_loss, g_pred = self.generator.pretrain_step(self.sess,
                                                                 x, class_label)
                supervised_g_losses.append(g_loss)
            # print results
            mean_g_loss = np.mean(supervised_g_losses)
            t_bar.set_postfix(G_loss=mean_g_loss)

        for class_label in range(0, self.CLASS_NUM):
            samples = self.generate_samples(self.SAMPLE_NUM, label_input=True, target_class=class_label)
            results = OrderedDict({'exp_name': self.PREFIX})
            results['class_label'] = class_label
            mm.compute_results(self.prior_classifier_fn,
                            samples, self.train_samples, self.ord_dict, results=results)
        self.mle_loader.create_batches(samples)

        if self.LAMBDA_C != 0:

            if self.verbose:
                print('\nDISCRIMINATOR PRETRAINING')
            t_bar = trange(self.PRETRAIN_DIS_EPOCHS)
            for i in t_bar:
                dis_x_train, dis_y_train = [], []   
                for class_label in range(0, self.CLASS_NUM):    
                    negative_samples = self.generate_samples(self.POSITIVE_NUM, label_input=True, target_class=class_label)
                    x1, y1 = self.dis_loader.load_train_data(
                        self.positive_samples, negative_samples)
                    dis_x_train.extend(x1)
                    dis_y_train.extend(y1)
                dis_batches = self.dis_loader.batch_iter(
                    zip(dis_x_train, dis_y_train), self.DIS_BATCH_SIZE,
                    self.PRETRAIN_DIS_EPOCHS)
                supervised_d_losses = []
                for batch in dis_batches:
                    x_batch, y_batch = zip(*batch)
                    x, x_label = zip(*x_batch)
                    # x_batch.size = (batch_size, sequence_length), y_batch.size = (batch_size, 2)
                    _, d_loss, _, _, _ = self.discriminator.train(
                        self.sess, x, y_batch, self.DIS_DROPOUT)

                    supervised_d_losses.append(d_loss)
                # print results
                mean_d_loss = np.mean(supervised_d_losses)
                t_bar.set_postfix(D_loss=mean_d_loss)
        
        self.PRETRAINED = True
        return

    def generate_samples(self, num, label_input=False, target_class=None):
        """Generates molecules.

        Arguments
        -----------
            - num. Integer number of molecules to generate
            - label_input. Boolean whether to use target class as input
            - target_class. Integer target class label
        """
        generated_samples = []
        
        for _ in range(int(num / self.GEN_BATCH_SIZE)):
            if target_class is False:
                for class_label in range(0, self.CLASS_NUM):
                    # tensor of class labels    
                    class_labels = [class_label] * self.GEN_BATCH_SIZE
                    samples = self.generator.generate(self.sess, class_labels, label_input)
                    # add generated samples and class labels
                    for i in range(self.GEN_BATCH_SIZE):
                        generated_samples.append([samples[i].tolist(), class_label])
            else:
                class_labels = [target_class] * self.GEN_BATCH_SIZE
                samples = self.generator.generate(self.sess, class_labels, label_input)
                for i in range(self.GEN_BATCH_SIZE):
                    generated_samples.append([samples[i].tolist(), target_class])

        return generated_samples

    def report_rewards(self, rewards, metric):
        print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
        print('Reward: {}  (lambda={:.2f})'.format(metric, self.LAMBDA))
        #np.set_printoptions(precision=3, suppress=True)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        min_r, max_r = np.min(rewards), np.max(rewards)
        print('Stats: {:.3f} ({:.3f}) [{:3f},{:.3f}]'.format(
            mean_r, std_r, min_r, max_r))
        non_neg = rewards[rewards > 0.01]
        if len(non_neg) > 0:
            mean_r, std_r = np.mean(non_neg), np.std(non_neg)
            min_r, max_r = np.min(non_neg), np.max(non_neg)
            print('Valid: {:.3f} ({:.3f}) [{:3f},{:.3f}]'.format(
                mean_r, std_r, min_r, max_r))
        #np.set_printoptions(precision=8, suppress=False)
        return
        
    def prior_classifier_fn(self, samples, class_labels, model_classifier):
        """Calculate rewards for generated samples using classifier
        
        Args:
            samples: generated samples
            class_labels: target class labels
            model_classifier: trained classifier model
        
        Returns:
            rewards: reward values array
        """       
        # Decode sequences as SMILES strings
        decoded = [mm.decode(sample, self.ord_dict) for sample in samples]    
        # print("decoded:", decoded)
        pct_unique = len(list(set(decoded))) / float(len(decoded))
        rewards = mm.batch_classifier(decoded, self.train_samples, class_labels)  
        length_scores = mm.batch_length(decoded, max_length=self.MAX_LENGTH-10, min_length=25)
                    
        weights = []
        for i, sample in enumerate(decoded):
            # 基础weight: 唯一性/重复度
            base_weight = pct_unique / float(decoded.count(sample))
            # 将length得分作为权重的调节因子
            length_weight = length_scores[i]
            # 组合weight
            combined_weight = base_weight * (0.2 + 0.8 * length_weight)  # 保证weight不会完全为0
            weights.append(combined_weight)
        
        weights = np.array(weights)              
        # pct_unique = len(list(set(decoded))) / float(len(decoded))
        # weights = np.array([pct_unique / float(decoded.count(sample)) for sample in decoded])
                
        return rewards * weights
    
    def report_classify_results(self, prior_classifier_fn, samples, class_labels, ord_dict):
        """report classify results
        
        Args:
            prior_classifier_fn: classify reward function
            samples: generated samples
            class_labels: target class labels
            ord_dict: character mapping dictionary
        """
        decoded = [mm.decode(sample, ord_dict) for sample in samples]
        
        valid_count = sum(1 for s in decoded if mm.verify_sequence(s))
        valid_ratio = valid_count / len(decoded)
        
        # Calculate classify accuracy
        rewards = prior_classifier_fn(samples, class_labels) 
        
        print('\nClassify results:')
        print('------------------------')
        print(f'Valid molecule ratio: {valid_ratio:.3f}')
        # print(f'Classify accuracy: {self.classify_accuracy:.3f}')
        print(f'Average reward: {np.mean(rewards):.3f}')
        print(f'Max reward: {np.max(rewards):.3f}')
        print(f'Min reward: {np.min(rewards):.3f}')
        print('------------------------\n')
        
        
    def conditional_train(self, ckpt_dir='checkpoints/', gen_steps=50):
        """Conditional training
        
        Args:
            ckpt_dir: checkpoint directory 
            gen_steps: number of generator steps
        """
        
        if not self.PRETRAINED and not self.SESS_LOADED:

            self.sess.run(tf.global_variables_initializer())
            self.pretrain()

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file = os.path.join(ckpt_dir,
                                     '{}_pretrain_ckpt'.format(self.PREFIX))
            saver = tf.train.Saver()
            path = saver.save(self.sess, ckpt_file)
            if self.verbose:
                print('Pretrain saved at {}'.format(path))
        
        if self.CLASS_NUM > 1 and not self.PRIOR_CLASSIFIER:
            prior_classifier(self.train_samples)
            print('\nClassifier training completed')
            self.PRIOR_CLASSIFIER = True

        if self.PRIOR_CLASSIFIER:
            from ADSeqGAN.piror_classifier import load_model
            classifier_model = load_model()                   
            def batch_reward(samples, train_samples=None):
                # Assign target class labels to each sample
                # class_labels = [class_label] * len(samples)  # class_idx is the current training generator class
                return self.prior_classifier_fn(samples, class_labels, classifier_model)            
            
        if not hasattr(self, 'class_rollout'):
            self.class_rollout = Rollout(self.generator, 0.3, self.PAD_NUM)
            
        if self.verbose:
            print('\nSTARTING TRAINING')
            print('============================\n')
        
        total_steps = gen_steps if gen_steps is not None else self.TOTAL_BATCH
        
        results_rows = []
        losses = defaultdict(list)
        t_bar = trange(total_steps)
        for nbatch in t_bar:
            
            results = OrderedDict({'exp_name': self.PREFIX})
            results['Batch'] = nbatch
            print('\nBatch n. {}'.format(nbatch))
            print('============================\n')
            print('\nGENERATOR TRAINING')
            print('============================\n')

            for class_label in range(0, self.CLASS_NUM):
                if nbatch % 10 == 0:
                    gen_samples = self.generate_samples(self.BIG_SAMPLE_NUM, 
                                                     label_input=True,
                                                     target_class=class_label)
                else:
                    gen_samples = self.generate_samples(self.SAMPLE_NUM,
                                                     label_input=True,
                                                     target_class=class_label)
                gen_molecules, class_labels = zip(*gen_samples)
                self.report_classify_results(
                    lambda x, y: self.prior_classifier_fn(x, y, classifier_model),
                    gen_molecules,
                    class_labels, 
                    self.ord_dict
                )
                
                results['class_label'] = class_label
                mm.compute_results(batch_reward,
                                   gen_samples, self.train_samples, self.ord_dict, results=results)
                
                for it in range(self.GEN_ITERATIONS):
                    class_labels = [class_label] * self.GEN_BATCH_SIZE
                    samples = self.generator.generate(self.sess, class_labels, label_input=True)
                    rewards = self.class_rollout.get_reward(
                        self.sess, samples, 24, self.discriminator,
                        batch_reward, self.LAMBDA_C)
                    g_loss = self.generator.generator_step(
                        self.sess, samples, rewards, class_labels)
                    losses['G-loss'].append(g_loss)
                    self.generator.g_count = self.generator.g_count + 1
                    self.report_rewards(rewards, 'classify')
                
            # t_bar.set_postfix(G_loss=np.mean(losses['G-loss']))
            self.class_rollout.update_params()

            if self.LAMBDA_C != 0:
                print('\nDISCRIMINATOR TRAINING')
                print('============================\n')
                for i in range(self.DIS_EPOCHS):
                    print('Discriminator epoch {}...'.format(i + 1))
                    
                    for class_label in range(0, self.CLASS_NUM):
                        negative_samples = self.generate_samples(self.POSITIVE_NUM, 
                                                                  label_input=True,
                                                                  target_class=class_label)
                        dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                            self.positive_samples, negative_samples)
                        dis_batches = self.dis_loader.batch_iter(
                            zip(dis_x_train, dis_y_train),
                            self.DIS_BATCH_SIZE, self.DIS_EPOCHS
                        )
                                            
                        d_losses, ce_losses, l2_losses, w_loss = [], [], [], []
                        for batch in dis_batches:
                            x_batch, y_batch = zip(*batch)
                            x_data, x_label = zip(*x_batch)
                            _, d_loss, ce_loss, l2_loss, w_loss = self.discriminator.train(
                                self.sess, x_data, y_batch, self.DIS_DROPOUT)
                            d_losses.append(d_loss)
                            ce_losses.append(ce_loss)
                            l2_losses.append(l2_loss)

                    losses['D-loss'].append(np.mean(d_losses))
                    losses['CE-loss'].append(np.mean(ce_losses))
                    losses['L2-loss'].append(np.mean(l2_losses))
                    losses['WGAN-loss'].append(np.mean(l2_losses))

                    self.discriminator.d_count = self.discriminator.d_count + 1

                print('\nDiscriminator trained.')
            
            results_rows.append(results)
                
            if nbatch % self.EPOCH_SAVES == 0 or nbatch == total_steps - 1:
                if results_rows is not None:
                    df = pd.DataFrame(results_rows)
                    df.to_csv('{}_results.csv'.format(
                        self.PREFIX), index=False)
                model_saver = tf.train.Saver()
                ckpt_file = os.path.join(
                    ckpt_dir,
                    f'{self.PREFIX}_{nbatch}.ckpt'
                )
                path = model_saver.save(self.sess, ckpt_file)
                print('\nModel saved at {}'.format(path))
            
        # save model
        model_saver = tf.train.Saver()
        ckpt_file = os.path.join(
            ckpt_dir,
            f'{self.PREFIX}_final.ckpt'
        )
        path = model_saver.save(self.sess, ckpt_file)
        print('\nModel saved at {}'.format(path))

        print('\n######### FINISHED #########')
        
    def output_samples(self, num_samples, output_dir='epoch_data/', 
                       label_input=False, target_class=None):
        """
        Output generated samples to a csv file
        """
        generated_samples = self.generate_samples(num_samples,
                                                   label_input=label_input,
                                                   target_class=target_class)
        decoded = [mm.decode(sample[0], self.ord_dict) for sample in generated_samples]
        verified_decoded = [s for s in decoded if mm.verify_sequence(s)]
        df = pd.DataFrame(verified_decoded, columns=['molecule'])
        df['class'] = target_class
        df.to_csv(f'{output_dir}/{self.PREFIX}_{target_class}_samples.csv', index=False)
        print(f'Generated samples saved to {output_dir}/{self.PREFIX}_{target_class}_samples.csv')
