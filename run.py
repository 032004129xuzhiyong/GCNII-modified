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
from mytool import tuner as mtuner
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
    SchedulerClass = tool.import_class(**args['scheduler_class_args'])
    scheduler = SchedulerClass(optimizer,**args['scheduler_args'])
    # warp scheduler
    def sche_func(epoch, lr, epoch_logs):
       scheduler.step(epoch_logs[args['scheduler_monitor']])
    scheduler_callback = mcallback.SchedulerWrapCallback(sche_func,True)
    callback_list.append(scheduler_callback)

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
        # mcallback.DfSaveCallback(**args['dfcallback_args']),
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


def train_times_and_get_mean_metric(repeat_args, first_train_logs):
    if repeat_args['tuner_n_repeats'] >= 1 and isinstance(repeat_args['tuner_n_repeats'],int):
        pass
    else:
        repeat_args['tuner_n_repeats'] = 1

    # flag tuner
    repeat_args['tuner_flag'] = False

    # Load data outside the scope of repeated experiments
    data = load_mat(**repeat_args['dataset_args'])

    # collect logs
    repeat_df_list = [pd.DataFrame(first_train_logs)]
    for rep_idx in range(repeat_args['tuner_n_repeats']-1):
        rep_logs = train_one_args(repeat_args,data)
        repeat_df_list.append(pd.DataFrame(rep_logs))

    # compute mean metric
    loss_metric_list = mtorch.History()
    for df in repeat_df_list:
        df_col_names = df.columns
        metric_dict = df.iloc[:,df_col_names.str.contains('metric')].max(axis=0).to_dict()
        loss_dict = df.iloc[:,df_col_names.str.contains('loss')].min(axis=0).to_dict()
        loss_metric_list.update(metric_dict)
        loss_metric_list.update(loss_dict)
    mean_dict = loss_metric_list.mean()  # {key: np_mean_value}
    return {k: float(v) for k, v in mean_dict.items()}


def train_with_besthp_and_save_config_and_history(best_conf):
    """
    保存两个数据： 最优配置(存储为yaml文件) 和  多次实验的过程数据(pd.DataFrame数据格式存储为多个csv文件)
    :param best_conf: dict
    :return:
        None
    """
    # flag tuner
    best_conf['tuner_flag'] = False

    best_dir = best_conf['best_trial_save_dir']
    dataset_name = tool.get_basename_split_ext(best_conf['dataset_args']['mat_path'])
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
    # get first logs
    args['trial'] = trial
    args['tuner_flag'] = True
    first_logs = train_one_args(args)
    # train other times and compute mean metric/loss
    mean_logs = train_times_and_get_mean_metric(args, first_logs)
    return mean_logs[args['tuner_monitor']]


def parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # public
    def add_public_argument(parser):
        parser.add_argument('--config-paths','-cps',
                            nargs='+',
                            required=True,
                            help='yaml config paths. e.g. config/3sources.yaml',
                            dest='config_paths')
        parser.add_argument('--change-args','-ca',
                            nargs='*',
                            default=None,
                            help='change args. e.g. dataset_args.topk=10 model_args.hid_dim=64',
                            dest='change_args')
        parser.add_argument('--quiet','-q',
                            action='store_true',
                            default=False,
                            help='whether to show logs')

    # tuner
    parser_tuner = subparsers.add_parser('tuner')
    add_public_argument(parser_tuner)
    parser_tuner.set_defaults(func=parser_tuner_func)

    # run
    parser_run = subparsers.add_parser('run')
    add_public_argument(parser_run)
    parser_run.set_defaults(func=parser_run_func)
    parser_run.add_argument('--run-times','-rt',
                            default=1,
                            type=int,
                            help='run times',
                            dest='run_times')
    parser_run.add_argument('--result-dir','-rd',
                            default='temp_result/',
                            type=str,
                            help='result dir',
                            dest='result_dir')

    # grid search
    parser_grid = subparsers.add_parser('grid')
    add_public_argument(parser_grid)
    parser_grid.set_defaults(func=parser_grid_func)
    parser_grid.add_argument('--grid-search-space','-gss',
                             nargs='*',
                             default=None,
                             help='grid search space. e.g. dataset_args.topk=[10,20,30] model_args.hid_dim=[64,128]',
                             dest='grid_search_space')

    args = parser.parse_args()
    args.func(args)


def parser_tuner_func(args):
    parser_args = vars(args)
    parser_args.pop('func')

    # get parser args
    expand_args = benedict()
    change_args = parser_args.pop('change_args')  # List
    for k, v in parser_args.items():
        expand_args[k] = v
    if change_args is not None:
        for change_arg in change_args:
            k, v = change_arg.split('=')
            expand_args[k] = eval(v)

    for conf in expand_args['config_paths']:
        yaml_args = benedict.from_yaml(conf)

        # update parser args
        expand_args = tool.remove_dict_None_value(expand_args)  # clean None value
        yaml_args.deepupdate(expand_args)

        # update callbacks save path
        yaml_args['dfcallback_args.df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(yaml_args['dataset_args']['mat_path']) + '.csv')
        yaml_args['tbwriter_args.log_dir'] = os.path.join('./logs/', tool.get_basename_split_ext(yaml_args['dataset_args']['mat_path']))
        yaml_args['earlystop_args.checkpoint_dir'] = os.path.join('./checkpoint/', tool.get_basename_split_ext(yaml_args['dataset_args']['mat_path']))

        # flag tuner
        yaml_args['tuner_flag'] = False

        args = yaml_args.dict()
        if tool.has_hyperparameter(args):
            # tuner
            args['tuner_flag'] = True

            if 'loss' in args['tuner_monitor']:
                direction = 'minimize'
            else:
                direction = 'maximize'
            study = optuna.create_study(direction=direction,
                                        study_name=tool.get_basename_split_ext(args['dataset_args']['mat_path']),
                                        storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                           heartbeat_interval=60,
                                                                           grace_period=120,
                                                                           failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                        load_if_exists=True,
                                        pruner=mtuner.CombinePruner([
                                            optuna.pruners.MedianPruner(n_warmup_steps=30),
                                            optuna.pruners.PercentilePruner(0.1, n_warmup_steps=30)
                                        ]),
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
            raise ValueError('No hyperparameter to tune!!')


def parser_run_func(args):
    parser_args = vars(args)
    parser_args.pop('func')

    # get parser args
    expand_args = benedict()
    change_args = parser_args.pop('change_args')  # List
    for k, v in parser_args.items():
        expand_args[k] = v
    if change_args is not None:
        for change_arg in change_args:
            k, v = change_arg.split('=')
            expand_args[k] = eval(v)

    for conf in expand_args['config_paths']:
        yaml_args = benedict.from_yaml(conf)

        # update parser args
        expand_args = tool.remove_dict_None_value(expand_args)  # clean None value
        yaml_args.deepupdate(expand_args)

        # update callbacks save path
        yaml_args['dfcallback_args.df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(yaml_args['dataset_args']['mat_path']) + '.csv')
        yaml_args['tbwriter_args.log_dir'] = os.path.join('./logs/', tool.get_basename_split_ext(yaml_args['dataset_args']['mat_path']))
        yaml_args['earlystop_args.checkpoint_dir'] = os.path.join('./checkpoint/', tool.get_basename_split_ext(yaml_args['dataset_args']['mat_path']))

        # flag tuner
        yaml_args['tuner_flag'] = False

        args = yaml_args.dict()
        if tool.has_hyperparameter(args):
            raise ValueError('Has hyperparameter!!')
        else:
            # only one config and train times
            best_args = args
            best_args['best_trial'] = parser_args['run_times']
            best_args['best_trial_save_dir'] = parser_args['result_dir']
            train_with_besthp_and_save_config_and_history(best_args)


def parser_grid_func(args):
    parser_args = vars(args)
    parser_args.pop('func')

    # get parser args
    expand_args = benedict()
    change_args = parser_args.pop('change_args')  # List
    grid_search_space = parser_args.pop('grid_search_space')  # List
    for k, v in parser_args.items():
        expand_args[k] = v
    if change_args is not None:
        for change_arg in change_args:
            k, v = change_arg.split('=')
            expand_args[k] = eval(v)
    # get grid search space
    # grid_search_space: List[str] e.g. ['dataset_args.topk=[10,20,30]','model_args.hid_dim=[64,128]']
    # in grid_search_space, key is full name, value is list
    # but in parser_grid_search_space, key is short name, value is list
    parser_grid_search_space = {}
    if grid_search_space is not None:
        for grid_search_arg in grid_search_space:
            k, v = grid_search_arg.split('=')
            parser_grid_search_space[k.split('.')[-1]] = eval(v)
    # else:
    #     raise ValueError('No grid search space!!')

    for conf in expand_args['config_paths']:
        # best/dataset/conf.yaml It has no hyperparameter to tune
        yaml_args = benedict.from_yaml(conf)

        # add hyperparameter to tune from grid_search_space
        # grid_search_space: List[str] e.g. ['dataset_args.topk=[10,20,30]','model_args.hid_dim=[64,128]']
        if grid_search_space is not None:
            for grid_search_arg in grid_search_space:
                k, v = grid_search_arg.split('=')
                # add something in location which will be searched in grid search.
                # It is a placehold, and will be replaced by grid search.
                v = eval(v)
                if type(v[0]) == int:
                    yaml_args[k] = {'type': 'int', 'low': 0, 'high': 10}
                elif type(v[0]) == float:
                    yaml_args[k] = {'type': 'float', 'low': 0.0, 'high': 10.0}
                elif type(v[0]) == str:
                    yaml_args[k] = {'type': 'categorical', 'choices': v}
        elif tool.has_hyperparameter(yaml_args.dict()):
            # maybe config has hyperparameter to tune
            # should transform hyperparameter to parser_grid_search_space
            pass
        else:
            raise ValueError('No hyperparameter to tune!!')

        # update parser args
        expand_args = tool.remove_dict_None_value(expand_args)
        yaml_args.deepupdate(expand_args)

        # flag tuner
        yaml_args['tuner_flag'] = False

        args = yaml_args.dict()
        if tool.has_hyperparameter(args):
            # tuner
            args['tuner_flag'] = True

            if 'loss' in args['tuner_monitor']:
                direction = 'minimize'
            else:
                direction = 'maximize'
            study = optuna.create_study(direction=direction,
                                        study_name='grid_'+tool.get_basename_split_ext(args['dataset_args']['mat_path']),
                                        storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                           heartbeat_interval=60,
                                                                           grace_period=120,
                                                                           failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
                                        load_if_exists=True,
                                        pruner=optuna.pruners.NopPruner(),
                                        sampler=optuna.samplers.GridSampler(parser_grid_search_space))
            study.optimize(lambda trial: objective(trial, args),
                           n_trials=99999,
                           gc_after_trial=True,
                           show_progress_bar=True,
                           callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])

        else:
            raise ValueError('No hyperparameter to tune!!')


if __name__ == '__main__':
    parser_args()


# def parser_args():
#     parser = argparse.ArgumentParser()
#     # config
#     parser.add_argument('--config-paths','-cps',
#                         nargs='+',
#                         required=True,
#                         help='yaml config paths. e.g. config/3sources.yaml',
#                         dest='config_paths')
#     # tuner
#     parser.add_argument('--tuner-monitor','-tm',
#                         default=None,
#                         type=str,
#                         help='tuner monitor metrics. e.g. loss/val_loss/metric_acc/val_metric_acc',
#                         dest='tuner_monitor')
#     parser.add_argument('--tuner-n-trials','-tn',
#                         default=None,
#                         type=int,
#                         help='tuner n trials',
#                         dest='tuner_n_trials')
#     parser.add_argument('--tuner-n-repeats','-tr',
#                         default=None,
#                         type=int,
#                         help='tuner n repeats',
#                         dest='tuner_n_repeats')
#     parser.add_argument('--best-trial','-bt',
#                         default=None,
#                         type=int,
#                         help='best trial',
#                         dest='best_trial')
#     parser.add_argument('--best-trial-save-dir','-btsd',
#                         default=None,
#                         type=str,
#                         help='best trial save dir',
#                         dest='best_trial_save_dir')
#     # dataset
#     parser.add_argument('--topk',
#                         default=None,
#                         type=int,
#                         help='knn topk',
#                         dest='dataset_args.topk')
#     parser.add_argument('--train-ratio',
#                         default=None,
#                         type=float,
#                         help='train val split ratio',
#                         dest='dataset_args.train_ratio')
#     # model args
#     parser.add_argument('--layer-class','-lc',
#                         default=None,
#                         type=str,
#                         help='GCNII layer class, all layer classes: GCNIILayer/GCNII_star_Layer',
#                         dest='model_args.layerclass')
#     parser.add_argument('--n-layer','-nl',
#                         default=None,
#                         type=int,
#                         help='number of layers',
#                         dest='model_args.nlayer')
#     parser.add_argument('--hid-dim','-hd',
#                         default=None,
#                         type=int,
#                         help='hidden dimension',
#                         dest='model_args.hid_dim')
#     parser.add_argument('--dropout','-dp',
#                         default=None,
#                         type=float,
#                         help='dropout ratio',
#                         dest='model_args.dropout')
#     parser.add_argument('--alpha','-al',
#                         default=None,
#                         type=float,
#                         help='alpha/h0',
#                         dest='model_args.alpha')
#     parser.add_argument('--lamda','-la',
#                         default=None,
#                         type=float,
#                         help='beta=lamda/l(nth-layer)',
#                         dest='model_args.lamda')
#     # training
#     parser.add_argument('--device','-dv',
#                         default=None,
#                         type=str,
#                         help='torch device')
#     parser.add_argument('--epochs','-ep',
#                         default=None,
#                         type=int,
#                         help='number of epochs')
#     parser.add_argument('--loss-weights','-lw',
#                         nargs='*',
#                         default=None,
#                         help='loss weights',
#                         dest='loss_weights')
#     parser.add_argument('--save-best-only','-sbo',
#                         action='store_true',
#                         default=None,
#                         help='earlystop args if save best weights only',
#                         dest='earlystop_args.save_best_only')
#     parser.add_argument('--monitor','-m',
#                         default=None,
#                         type=str,
#                         help='earlystop args monitor metrics. e.g. loss/val_loss/metric_acc/val_metric_acc',
#                         dest='earlystop_args.monitor')
#     parser.add_argument('--patience','-pt',
#                         default=None,
#                         type=int,
#                         help='earlystop args patience',
#                         dest='earlystop_args.patience')
#     parser.add_argument('--quiet','-q',
#                         action='store_true',
#                         default=False,
#                         help='whether to show logs')
#     # others
#     parser.add_argument('--train-times-with-no-tuner', '-ttnt',
#                         default=1,
#                         type=int,
#                         help='训练实验次数，没有超参数搜索(默认有超参数搜索)',
#                         dest='tt_nt')
#     parser.add_argument('--train-save-dir-with-no-tuner', '-tsdnt',
#                         default='temp_result/',
#                         type=str,
#                         help='训练实验数据保存目录，没有超参数搜索(默认有超参数搜索)',
#                         dest='tsd_nt')
#     return parser.parse_args()

# if __name__ == '__main__':
#     parser_args = vars(parser_args())
#     expand_args = benedict()
#     for k,v in parser_args.items():
#         expand_args[k] = v
#
#     for conf in expand_args['config_paths']:
#         yaml_args = benedict.from_yaml(conf)
#
#         # update parser args
#         expand_args = tool.remove_dict_None_value(expand_args) # clean None value
#         yaml_args.deepupdate(expand_args)
#
#         # update callbacks save path
#         yaml_args['dfcallback_args.df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(conf) + '.csv')
#         yaml_args['tbwriter_args.log_dir'] = os.path.join('./logs/', tool.get_basename_split_ext(conf))
#         yaml_args['earlystop_args.checkpoint_dir'] = os.path.join('./checkpoint/', tool.get_basename_split_ext(conf))
#
#         # flag tuner
#         yaml_args['tuner_flag'] = False
#
#         args = yaml_args.dict()
#         if tool.has_hyperparameter(args):
#             # tuner
#             args['tuner_flag'] = True
#
#             if 'loss' in args['tuner_monitor']:
#                 direction = 'minimize'
#             else:
#                 direction = 'maximize'
#             study = optuna.create_study(direction=direction,
#                                         study_name=tool.get_basename_split_ext(conf),
#                                         storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
#                                                                            heartbeat_interval=60,
#                                                                            grace_period=120,
#                                                                            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3)),
#                                         load_if_exists=True,
#                                         pruner=mtuner.CombinePruner([
#                                             optuna.pruners.MedianPruner(n_warmup_steps=30),
#                                             optuna.pruners.PercentilePruner(0.1,n_warmup_steps=30)
#                                         ]),
#                                         sampler=optuna.samplers.TPESampler())
#             study.optimize(lambda trial: objective(trial, args),
#                            n_trials=args['tuner_n_trials'],
#                            gc_after_trial=True,
#                            show_progress_bar=True,
#                            callbacks=[mcallback.StudyStopWhenTrialKeepBeingPrunedCallback(20)])
#
#             # get best args
#             best_hp = study.best_params
#             # print
#             for i in range(5):
#                 print('*' * 50)
#             tool.print_dicts_tablefmt([best_hp], ['Best HyperParameters!!'])
#             for i in range(5):
#                 print('*' * 50)
#
#             # train times with best args
#             best_args = tool.modify_dict_with_trial(args, study.best_trial)
#             train_with_besthp_and_save_config_and_history(best_args)
#         else:
#             # only one config and train times
#             best_args = args
#             best_args['best_trial'] = parser_args['tt_nt']
#             best_args['best_trial_save_dir'] = parser_args['tsd_nt']
#             train_with_besthp_and_save_config_and_history(best_args)
#