import numpy as np
import matplotlib.pyplot as plt


from fairlearn.postprocessing._threshold_optimizer import _reformat_and_group_data
from fairlearn.postprocessing._roc_curve_utilities import _calculate_roc_points
from fairlearn.postprocessing._roc_curve_utilities import _get_roc, _interpolate_curve

plt.rcParams['figure.figsize'] = (12, 5)

def plot_perc_true_sex(y_true, sex):

    def percentage_with_label_1(sex_value):
        return y_true[sex == sex_value].sum() / (sex == sex_value).sum()


    plt.bar([0, 1],
            [percentage_with_label_1("female"), percentage_with_label_1("male")],
            color='g')
    plt.xticks([0, 1], ["female", "male"])
    plt.ylabel("percentage earning over $50,000")
    plt.xlabel("sex")
    plt.show()

def get_roc_points(data_grouped_by_sensitive_feature):
    roc_points = {}
    for group_name, group in data_grouped_by_sensitive_feature:
        roc_points[group_name] = _calculate_roc_points(
            data_grouped_by_sensitive_feature.get_group(group_name), 0
        )
    return roc_points


def plot_roc(data_grouped_by_sensitive_feature, roc_points, **kwargs):
    for group_name, group in data_grouped_by_sensitive_feature:
        plt.plot(roc_points[group_name].x, roc_points[group_name].y, label=group_name)

    plt.xlabel("$P [ \\hat{Y}=1 | Y=0 ]$")
    plt.ylabel("$P [ \\hat{Y}=1 | Y=1 ]$")
    plt.legend()

def calc_curves(sensitive_feature, y_true, scores, grid):
    data_grouped_by_sensitive_feature = _reformat_and_group_data(sensitive_feature, y_true, scores)
    n = len(y_true)

    overall_tradeoff_curve = 0
    selection_error_curves = {}
    roc_convex_hulls = {}
    roc_unfiltered = {}
    for name, group in data_grouped_by_sensitive_feature:
        n_group = len(group)
        n_positive = sum(group['label'])
        n_negative = n_group - n_positive
        p_sensitive_feature_value = n_group / n

        roc_unfiltered[name] = _calculate_roc_points(group, name, flip=False)
        roc_convex_hulls[name] = _get_roc(group, name, flip=False)

        fraction_negative_label_positive_sample = (n_negative / n_group) * roc_convex_hulls[name]['x']
        fraction_positive_label_positive_sample = (n_positive / n_group) * roc_convex_hulls[name]['y']

        # Calculate selection to represent the proportion of positive predictions.
        roc_convex_hulls[name]['selection'] = fraction_negative_label_positive_sample + \
                                       fraction_positive_label_positive_sample

        fraction_positive_label_negative_sample = \
            (n_positive / n_group) * (1 - roc_convex_hulls[name]['y'])
        roc_convex_hulls[name]['error'] = fraction_negative_label_positive_sample + \
                                   fraction_positive_label_negative_sample

        selection_error_curves[name] = \
            _interpolate_curve(roc_convex_hulls[name], 'selection', 'error', 'operation', grid)

        overall_tradeoff_curve += p_sensitive_feature_value * \
                                  selection_error_curves[name]['error']

    return roc_unfiltered, roc_convex_hulls, selection_error_curves, overall_tradeoff_curve

def plot_creation_convex_hull(sensitive_feature, y_true, scores):
    grid = np.linspace(0, 1, 100 + 1)
    roc_unfiltered, roc_convex_hulls, _, _ = \
        calc_curves(sensitive_feature, y_true, scores, grid)

    for name, selection_error_curve in roc_convex_hulls.items():

        plt.plot(roc_unfiltered[name]['x'],
                 roc_unfiltered[name]['y'],
                 marker='.',
                 linestyle='',
                 label='data',
                 color='black'
                 )

        plt.plot(roc_convex_hulls[name]['x'],
                 roc_convex_hulls[name]['y'],
                 marker='*',
                 linestyle='',
                 label='convex hull'
                 )

        plt.xlabel('False positive')
        plt.ylabel('True positive')
        plt.title(name)
        plt.legend()
        plt.show()








def plot_convex_hull_interpolation(sensitive_feature, y_true, scores):
    grid = np.linspace(0, 1, 100 + 1)
    _, roc_convex_hulls, selection_error_curves, _ = \
        calc_curves(sensitive_feature, y_true, scores, grid)

    for name, selection_error_curve in selection_error_curves.items():

        plt.plot(selection_error_curves[name]['selection'],
                 selection_error_curves[name]['error'],
                 marker='.',
                 linestyle='',
                 label='interpolated grid',
                 color='black'
                 )

        plt.plot(roc_convex_hulls[name]['selection'],
                 roc_convex_hulls[name]['error'],
                 marker='*',
                 linestyle='',
                 label='convex hull'
                 )

        plt.xlabel('selection')
        plt.ylabel('error')
        plt.title(name)
        plt.legend()
        plt.show()


def plot_overall_tradeoff(sensitive_feature, y_true, scores):
    grid = np.linspace(0, 1, 100 + 1)
    _, _, _, overall_tradeoff_curve = \
        calc_curves(sensitive_feature, y_true, scores, grid)

    plt.plot(grid, overall_tradeoff_curve)
    plt.xlabel('Selection')
    plt.ylabel('Error')
    plt.title('Trade off')
    plt.show()