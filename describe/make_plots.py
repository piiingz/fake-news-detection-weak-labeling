import pandas as pd
import matplotlib.pyplot as plt
from handle_datasets.paths import SUBSET_BOXPLOT_FLIERS_PATH, SUBSET_BOXPLOT_PATH, SUBSET_HISTOGRAM_PATH, BOXPLOT_PATH
import seaborn as sns


def subset_make_box_plots(subset: pd.DataFrame):
    subset = subset.select_dtypes('number')
    sns.set(rc={'figure.figsize': (6, 8)})

    for col in subset.columns:

        if col != 'label':

            plt.figure()
            sns.boxplot(x="label", y=col, data=subset, showfliers=False)
            plt.title(col)
            plt.savefig('{}/{}.png'.format(SUBSET_BOXPLOT_FLIERS_PATH, col))
            plt.close()

            plt.figure()
            sns.boxplot(x="label", y=col, data=subset)
            plt.title(col)
            plt.savefig('{}/{}.png'.format(SUBSET_BOXPLOT_PATH, col))
            plt.close()


def make_box_plots(subsets: list):

    total = pd.concat(subsets, ignore_index=True)

    for col in total.columns:

        if col != 'label':

            plt.figure()
            sns.boxplot(x="label", y=col, data=total)
            plt.title(col)
            plt.savefig('{}/{}.png'.format(BOXPLOT_PATH, col))
            plt.close()


def subset_histogram(subset: pd.DataFrame):
    sns.set(rc={'figure.figsize': (10, 4)})

    fake_features = subset.loc[subset.label == 1, (subset.dtypes == 'float64') | (subset.dtypes == 'int')]
    real_features = subset.loc[subset.label == 0, (subset.dtypes == 'float64') | (subset.dtypes == 'int')]

    for col in fake_features.columns:

        if col != 'label':

            plt.figure()

            sns.kdeplot(data=fake_features, x=col, shade=True, color='red', alpha=0.6, label="fake")
            sns.kdeplot(data=real_features, x=col, shade=True, color='blue', alpha=0.6, label="real")

            plt.legend()
            plt.title(col)
            plt.savefig('{}/{}.png'.format(SUBSET_HISTOGRAM_PATH, col))
            plt.close()
