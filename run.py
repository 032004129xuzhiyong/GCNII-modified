# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年11月23日
"""

import os
import argparse
from benedict import benedict
from mytool import tool


def bind_boolind_for_fn(func, train_bool_ind, val_bool_ind):

    def binded_func(scores, labels):
        if scores.requires_grad == True:
            return func(scores[train_bool_ind], labels[train_bool_ind])
        else:
            return func(scores[val_bool_ind], labels[val_bool_ind])

    tool.set_func_name(binded_func, tool.get_func_name(func))
    return binded_func




if __name__ == '__main__':
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

    parser_args = vars(parser.parse_args())
    expand_args = benedict()
    for k,v in parser_args.items():
        expand_args[k] = v

    for conf in expand_args['config_paths']:
        yaml_args = benedict.from_yaml(conf)

        # update parser args
        expand_args = expand_args.flatten('*').filter(lambda k,v: v is not None).unflatten('*') # clean None
        yaml_args.deepupdate(expand_args)

        # update callbacks save path
        yaml_args['dfcallback_args.df_save_path'] = os.path.join('./tables/', tool.get_basename_split_ext(conf) + '.csv')
        yaml_args['tbwriter_args.log_dir'] = os.path.join('./logs/', tool.get_basename_split_ext(conf))
        yaml_args['earlystop_args.checkpoint_dir'] = os.path.join('./checkpoint/', tool.get_basename_split_ext(conf))