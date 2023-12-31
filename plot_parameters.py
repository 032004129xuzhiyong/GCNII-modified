# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年11月26日
"""

from mytool import plot as mplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import os


def plot_dataset_to_df(dataset_to_df, layout=None,
                       field_check='val_metric_', plot_type='line',
                       files_dir='./parameters', show=False):
    """
    plot dataset_to_df and save to files_dir
    :param dataset_to_df: Dict[dataset_name,df]
    :param layout: (row,col)
    :param field_check: string to check in column name as y-axis-data
    :param plot_type: 'line' or 'bar'
    :param files_dir: string
    :param show: bool
    :return:
        None
    """
    num_dataset = len(dataset_to_df.keys())
    if layout is None:
        layout = (1, num_dataset)

    # set figure layout
    fig, axes = plt.subplots(nrows=layout[0], ncols=layout[1], figsize=(layout[1] * 5, layout[0] * 5))
    if layout[0] == 1 and layout[1] == 1:
        axes = [axes]
    elif layout[0] > 1 and layout[1] > 1:
        axes = axes.flatten()

    # plot
    for i, dataset_name in enumerate(dataset_to_df.keys()):
        df = dataset_to_df[dataset_name]
        # filter field_check
        filter_df = df.iloc[:, df.columns.str.contains(field_check)]
        # change column name
        new_filter_df_columns = filter_df.columns.str.replace(field_check, '')
        filter_df.columns = new_filter_df_columns
        # reset x-axis-data and index
        filter_df.index = df.iloc[:, 0]
        filter_df.reset_index(inplace=True)
        # rescale y-axis-data to [0,100]
        if 'metric' in field_check:
            filter_df.iloc[:, 1:] = filter_df.iloc[:, 1:] * 100
        if plot_type == 'line':
            ax: plt.Axes = mplot.plot_lines_with_compare_data(ax=axes[i], x=filter_df,
                                                              xlabel=filter_df.columns[0],
                                                              ylabel='Evaluation Metrics(%)',
                                                              title=dataset_name,
                                                              markevery=4)
        elif plot_type == 'bar':
            ax = mplot.plot_bars_with_compare_data(ax=axes[i], x=filter_df,
                                                   xlabel=filter_df.columns[0],
                                                   ylabel='Evaluation Metrics(%)',
                                                   title=dataset_name)

        # set y-lim to [0,100]
        ax.set_ylim(0, 100)
        # set legend
        ax.legend()
    plt.tight_layout(pad=2, h_pad=5, w_pad=5)
    plt.savefig(f'./{files_dir}/{filter_df.columns[0]}.png')
    if show:
        plt.show()


def make_csv_files_and_pngs(parameters, dataset_names, files_dir='./parameters'):
    """
    make csv files and pngs
    :param parameters: List[str]
    :param dataset_names: List[str]
    :param files_dir: str
    :return:
        None
    """
    for para in parameters:
        # new figure per parameter

        # first get data from optuna/sqlite.db
        # get data which has para as parameter per dataset
        dataset_to_df = {}
        for dataset_name in dataset_names:
            study = optuna.create_study(study_name=f'grid_{dataset_name}',
                                        storage=optuna.storages.RDBStorage('sqlite:///./tuner.db',
                                                                           heartbeat_interval=60,
                                                                           grace_period=120,
                                                                           failed_trial_callback=optuna.storages.RetryFailedTrialCallback(
                                                                               3)),
                                        load_if_exists=True,
                                        pruner=optuna.pruners.NopPruner())
            # filter trial and get user_attrs['mean_logs'] as data
            trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
            filtered_trials = [trial for trial in trials if
                               para in trial.params.keys() and len(trial.params.keys()) == 1]
            # List[int|str|float]
            trial_para_list = [trial.params[para] for trial in filtered_trials]
            # List[Dict[str, float]]
            trial_data_list = [trial.user_attrs['mean_logs'] for trial in filtered_trials]
            # Eliminate duplicate values
            el_trial_para_list, ind = np.unique(trial_para_list, return_index=True)
            el_trial_data_list = [trial_data_list[i] for i in ind]
            # Sort by el_trial_para_list
            arg_sort = np.argsort(el_trial_para_list)
            sort_el_trial_para_list = [el_trial_para_list[i] for i in arg_sort]
            sort_el_trial_data_list = [el_trial_data_list[i] for i in arg_sort]
            # make df
            df = pd.DataFrame(data=sort_el_trial_data_list, index=sort_el_trial_para_list)
            df.reset_index(inplace=True)
            df.rename(columns={'index': para}, inplace=True)
            # save df
            dataset_to_df[dataset_name] = df
            df.to_csv(f'./{files_dir}/{para}_{dataset_name}.csv', index=False)
        plot_dataset_to_df(dataset_to_df, layout=(1, 2))


def read_csv_files_to_dataset_to_df(parameter, dataset_names, files_dir='./parameters'):
    """
    read csv files to dataset_to_df
    :param parameter: string
    :param dataset_names: List[string]
    :param files_dir: string
    :return:
        dataset_to_df: Dict[dataset_name,df]
    """
    dataset_to_df = {}
    for dataset_name in dataset_names:
        df = pd.read_csv(f'{files_dir}/{parameter}_{dataset_name}.csv')
        dataset_to_df[dataset_name] = df
    return dataset_to_df


def plot_existing_csv_files(parameters, dataset_names, files_dir='./parameters'):
    """
    plot existing csv files and save to files_dir
    :param parameters: List[str]
    :param dataset_names: List[str]
    :param files_dir: str
    :return:
        None
    """
    for para in parameters:
        dataset_to_df = read_csv_files_to_dataset_to_df(para, dataset_names, files_dir)
        plot_dataset_to_df(dataset_to_df, layout=(1, 2))


if __name__ == '__main__':
    dataset_names = ['3sources', 'NGs', ]
    parameters = ['hid_dim', 'topk']
    files_dir = './parameters'

    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    #make_csv_files_and_pngs(parameters, dataset_names, files_dir)
    plot_existing_csv_files(parameters, dataset_names, files_dir)