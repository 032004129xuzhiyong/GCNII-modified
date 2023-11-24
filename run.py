# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年11月23日
"""
import copy
import os
import argparse
import optuna
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from benedict import benedict
from mytool import tool
from mytool import mytorch as mtorch
from mytool import callback as mcallback
from mytool import metric as mmetric
from mytool import plot as mplot
from datasets.dataset import load_mat
from typing import *


def bind_boolind_for_fn(func, train_bool_ind, val_bool_ind):

    def binded_func(scores, labels):
        if scores.requires_grad == True:
            return func(scores[train_bool_ind], labels[train_bool_ind])
        else:
            return func(scores[val_bool_ind], labels[val_bool_ind])

    tool.set_func_name(binded_func, tool.get_func_name(func))
    return binded_func


def train_one_args(args, data=None):
    # load data
    if data is not None:
        adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class = data
    else:
        adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class = load_mat(**args['dataset_args'])
    dataload = [((inputs, adjs), labels)]

    # build model and init callback_list
    device = args['device']
    if device == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    ModelClass = tool.import_class(**args['model_class_args'])
    model = ModelClass(n_feats=sum(n_feats), n_class=n_class, **args['model_args'])
    wrapmodel = mtorch.WrapModel(model).to(device)
    callback_list: List[mcallback.Callback] = [mcallback.PruningCallback(args['trial'],args['tuner_monitor'])] if args['tuner_flag'] else []

    # loss optimizer lr_scheduler
    loss_fn = nn.CrossEntropyLoss()
    OptimizerClass = tool.import_class(**args['optimizer_class_args'])
    optimizer = OptimizerClass([
        dict(params=model.reg_params, weight_decay=args['weight_decay1']),
        dict(params=model.non_reg_params, weight_decay=args['weight_decay2'])
    ], **args['optimizer_args'])
    # SchedulerClass = tool.import_class(**args['scheduler_class_args'])
    # scheduler = SchedulerClass(optimizer,**args['scheduler_args'])
    # # warp scheduler
    # def sche_func(epoch, lr, epoch_logs):
    #    scheduler.step(epoch_logs[args['scheduler_monitor']])
    # scheduler_callback = mcallback.SchedulerWrapCallback(sche_func,True)
    # callback_list.append(scheduler_callback)

    # training
    wrapmodel.compile(
        loss=bind_boolind_for_fn(loss_fn, train_bool, val_bool),
        optimizer=optimizer,
        metric=[
            bind_boolind_for_fn(mmetric.acc, train_bool, val_bool),
            bind_boolind_for_fn(mmetric.f1, train_bool, val_bool),
            bind_boolind_for_fn(mmetric.precision,train_bool,val_bool),
            bind_boolind_for_fn(mmetric.recall,train_bool,val_bool),
        ]
    )

    # add callbacks
    callback_list.extend([
        mcallback.DfSaveCallback(**args['dfcallback_args']),
        mcallback.EarlyStoppingCallback(quiet=args['quiet'],**args['earlystop_args']),
        mcallback.TunerRemovePreFileInDir([
            args['earlystop_args']['checkpoint_dir'],
        ],10,0.8),
    ])

    # fit
    history = wrapmodel.fit(
        dataload=dataload,
        epochs=args['epochs'],
        device=device,
        val_dataload=dataload,
        callbacks=callback_list,
        quiet=args['quiet']
    )

    return history.history


def train_with_besthp_and_save_config_and_history(best_conf):
    """
    保存两个数据： 最优配置(存储为yaml文件) 和  多次实验的过程数据(pd.DataFrame数据格式存储为多个csv文件)
    :param best_conf: dict
    :return:
        None
    """
    # flag tuner
    yaml_args['tuner_flag'] = False

    best_dir = best_conf['best_trial_save_dir']
    dataset_name = tool.get_basename_split_ext(best_conf['dfcallback_args']['df_save_path'])
    best_dataset_dir = os.path.join(best_dir, dataset_name)
    if not os.path.exists(best_dataset_dir):
        os.makedirs(best_dataset_dir)

    # Load data outside the scope of repeated experiments
    data = load_mat(**best_conf['dataset_args'])

    for tri_idx in range(best_conf['best_trial']):
        tri_logs = train_one_args(best_conf,data)
        df = pd.DataFrame(tri_logs)
        #csv
        df_save_path = os.path.join(best_dataset_dir, 'df' + str(tri_idx) + '.csv')
        df.to_csv(df_save_path, index=False, header=True)
    # {key1: [mean, std], key2: [mean, std]...}
    mean_std_metric_dict = compute_mean_metric_in_bestdir_for_one_dataset(best_dataset_dir, if_plot_fig=True)
    # {key1:{mean:float,std:float}, key2:{mean:float,std:float}...}
    mean_std_metric_dict = {key: {'mean':mean_std_metric_dict[key][0],'std':mean_std_metric_dict[key][1]}
                            for key in mean_std_metric_dict.keys()}
    best_conf.update(mean_std_metric_dict)
    save_conf_path = os.path.join(best_dataset_dir, 'conf.yaml')
    tool.save_yaml_args(save_conf_path, best_conf)


def compute_mean_metric_in_bestdir_for_one_dataset(one_dataset_dir, if_plot_fig=False):
    """

    Args:
        one_dataset_dir: str
        if_plot_fig: a plot every df

    Returns:
        mean_std_metric_dict: Dict {key1:[mean,std],key2:[mean,std]...}
    """
    filenames = os.listdir(one_dataset_dir)
    filenames = [name for name in filenames if name.endswith('csv')]
    filepaths = [os.path.join(one_dataset_dir, name) for name in filenames]

    metric_list = mtorch.History() # {key1:[], key2:[]...}
    for fp in filepaths:
        df = pd.read_csv(fp)
        df_col_names = df.columns
        # png
        if if_plot_fig:
            fig = mplot.plot_LossMetricTimeLr_with_df(df)
            fig.savefig(os.path.join(one_dataset_dir, tool.get_basename_split_ext(fp) + '.png'))
            plt.close()  # 关闭figure
            del fig
        metric_dict = df.iloc[:,df_col_names.str.contains('metric')].max(axis=0).to_dict()
        metric_list.update(metric_dict)
    metric_list = metric_list.history
    # {key1:[mean,std],key2:[mean,std]...}
    mean_std_metric_dict = {key:[float(np.mean(metric_list[key])), float(np.std(metric_list[key]))]
                            for key in metric_list.keys()}
    return mean_std_metric_dict


def compute_mean_metric_in_bestdir_for_all_dataset(best_dir):
    """
    计算best目录下所有数据集 mean_acc 和 std_acc
    :param best_dir:
    :return:
        Dict[datasetname, (mean_acc, std_acc)]
    """
    dataset_names = os.listdir(best_dir)
    dataset_dirs = [os.path.join(best_dir, dn) for dn in dataset_names]
    dataset_mean_std = [compute_mean_metric_in_bestdir_for_one_dataset(ddir) for ddir in dataset_dirs]
    return dict(zip(dataset_names, dataset_mean_std))


def objective(trial: optuna.trial.Trial, extra_args):
    args = copy.deepcopy(extra_args)
    args = tool.modify_dict_with_trial(args, trial)
    args['trial'] = trial
    args['tuner_flag'] = True
    # get history epoch
    history = train_one_args(args)
    if 'loss' in args['tuner_monitor']:
        return min(history[args['tuner_monitor']])
    else:  # metric
        return max(history[args['tuner_monitor']])


def parser_args():
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument('--config-paths','-cps',
                        nargs='+',
                        required=True,
                        help='yaml config paths. e.g. config/3sources.yaml',
                        dest='config_paths')
    # dataset
    parser.add_argument('--topk',
                        default=None,
                        type=int,
                        help='knn topk',
                        dest='dataset_args.topk')
    parser.add_argument('--train-ratio','-tr',
                        default=None,
                        type=float,
                        help='train val split ratio',
                        dest='dataset_args.train_ratio')
    # model args
    parser.add_argument('--layer-class','-lc',
                        default=None,
                        type=str,
                        help='GCNII layer class, all layer classes: GCNIILayer/GCNII_star_Layer',
                        dest='model_args.layerclass')
    parser.add_argument('--n-layer','-nl',
                        default=None,
                        type=int,
                        help='number of layers',
                        dest='model_args.nlayer')
    parser.add_argument('--hid-dim','-hd',
                        default=None,
                        type=int,
                        help='hidden dimension',
                        dest='model_args.hid_dim')
    parser.add_argument('--dropout','-dp',
                        default=None,
                        type=float,
                        help='dropout ratio',
                        dest='model_args.dropout')
    parser.add_argument('--alpha','-al',
                        default=None,
                        type=float,
                        help='alpha/h0',
                        dest='model_args.alpha')
    parser.add_argument('--lamda','-la',
                        default=None,
                        type=float,
                        help='beta=lamda/l(nth-layer)',
                        dest='model_args.lamda')
    # training
    parser.add_argument('--device','-dv',
                        default=None,
                        type=str,
                        help='torch device')
    parser.add_argument('--epochs','-ep',
                        default=None,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--loss-weights','-lw',
                        nargs='*',
                        default=None,
                        help='loss weights',
                        dest='loss_weights')
    parser.add_argument('--save-best-only','-sbo',
                        action='store_true',
                        default=None,
                        help='earlystop args if save best weights only',
                        dest='earlystop_args.save_best_only')
    parser.add_argument('--monitor','-m',
                        default=None,
                        type=str,
                        help='earlystop args monitor metrics. e.g. loss/val_loss/metric_acc/val_metric_acc',
                        dest='earlystop_args.monitor')
    parser.add_argument('--patience','-pt',
                        default=None,
                        type=int,
                        help='earlystop args patience',
                        dest='earlystop_args.patience')
    parser.add_argument('--quiet','-q',
                        action='store_true',
                        default=False,
                        help='whether to show logs')
    # others
    parser.add_argument('--train-times-with-no-tuner',
                        default=1,
                        type=int,
                        help='训练实验次数，没有超参数搜索(默认有超参数搜索)',
                        dest='tt_nt')
    parser.add_argument('--train-save-dir-with-no-tuner',
                        default='temp_result/',
                        type=str,
                        help='训练实验数据保存目录，没有超参数搜索(默认有超参数搜索)',
                        dest='tsd_nt')
    return parser.parse_args()


if __name__ == '__main__':
    parser_args = vars(parser_args())
    expand_args = benedict()
    for k,v in parser_args.items():
        expand_args[k] = v

    for conf in expand_args['config_paths']:
        yaml_args = benedict.from_yaml(conf)

        # update parser args
        expand_args = tool.remove_dict_None_value(expand_args) # clean None value
        yaml_args.deepupdate(expand_args)

        # update callbacks save path
        yaml_args['dfcallback_args.df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(conf) + '.csv')
        yaml_args['tbwriter_args.log_dir'] = os.path.join('./logs/', tool.get_basename_split_ext(conf))
        yaml_args['earlystop_args.checkpoint_dir'] = os.path.join('./checkpoint/', tool.get_basename_split_ext(conf))

        # flag tuner
        yaml_args['tuner_flag'] = False

        args = yaml_args.dict()
        if tool.has_hyperparameter(args):
            # tuner
            yaml_args['tuner_flag'] = True

            if 'loss' in args['tuner_monitor']:
                direction = 'minimize'
            else:
                direction = 'maximize'
            study = optuna.create_study(direction=direction,
                                        study_name=tool.get_basename_split_ext(conf),
                                        storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                           heartbeat_interval=60,
                                                                           grace_period=120,
                                                                           failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                        load_if_exists=True,
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
                                        sampler=optuna.samplers.TPESampler())
            study.optimize(lambda trial: objective(trial, args),
                           n_trials=args['tuner_n_trials'],
                           gc_after_trial=True,
                           show_progress_bar=True,
                           callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])

            # get best args
            best_hp = study.best_params
            # print
            for i in range(5):
                print('*' * 50)
            tool.print_dicts_tablefmt([best_hp], ['Best HyperParameters!!'])
            for i in range(5):
                print('*' * 50)

            # train times with best args
            best_args = tool.modify_dict_with_trial(args, study.best_trial)
            train_with_besthp_and_save_config_and_history(best_args)
        else:
            # only one config and train times
            best_args = args
            best_args['best_trial'] = parser_args['tt_nt']
            best_args['best_trial_save_dir'] = parser_args['tsd_nt']
            train_with_besthp_and_save_config_and_history(best_args)