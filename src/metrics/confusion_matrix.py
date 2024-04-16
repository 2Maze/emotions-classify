from torchmetrics.classification import MulticlassConfusionMatrix
from torch import Tensor
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from torch.nn import Module

# import tensorflow as tf
class CreateConfMatrix(Module):
    def __init__(self, num_classes: int, lables: list[str], tensorboard_logger):
        super().__init__()
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.num_classes = num_classes
        self.lables = lables
        # self.tensorboard_logger = tensorboard_logger

    def draw_confusion_matrix(self, y_pred: Tensor, y_true: Tensor, epoch: int, tensorboard_logger, metric_name="Confusion matrix") -> None:
        figsize = (10, 8)
        # print(y_pred)
        # print(y_true)
        confusion_matrix = self.confusion_matrix(y_pred, y_true).cpu()

        df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        # plt.figure(figsize=figsize)
        fontsize_ticks = 36
        fontsize_ax = 18
        fontsize_title = 24
        fontsize_annotation = 18
        plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        plt.title('Confusion Matrix', fontsize=fontsize_title)
        # Show all ticks and label them with the respective list entries
        font = {
            # 'family': 'normal',
            #     'weight': 'bold',
                'size': 24}
        plt.rc('font', **font)

        hmap = sns.heatmap(df_cm, ax=ax, annot=True,  square=True,
                           fmt=".3g",  cmap='Spectral',
                           # xticklabels=self.lables,
                           # vmin=1, vmax=2,
                           annot_kws={'size':str(fontsize_annotation)},

                   # yticklabels=self.lables,
                           )
        plt.ylabel('Real', fontsize=fontsize_ax).set_rotation(90)
        plt.xlabel('Predicted', fontsize=fontsize_ax).set_rotation(0)
        ax.set_xticks(np.arange(len(self.lables)), labels=self.lables)
        ax.set_yticks(np.arange(len(self.lables)), labels=self.lables)
        # hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=fontsize_ticks, rotation=90, fontweight=100)
        # hmap.set_yticklabels(hmap.get_xmajorticklabels(), fontsize=fontsize_ticks, rotation=0, fontweight=100)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")
        fig.tight_layout()
        plt.close(fig)

        # (evaluate_streaming_metrics_op, reset_streaming_metrics_op, summary_op,
        # confusion_image_summary, confusion_image_placeholder, confusion_image) = self.add_evaluation_step(y_pred, y_true, self.num_classes)

        # res_fig = self.confusion_matrix_to_image_summary(confusion_matrix.numpy(), )

        tensorboard_logger.experiment.add_figure(metric_name, fig, epoch)


# Implementing methods to pass plotted images to summaries
# (copied from https://stackoverflow.com/a/49926139/23886223)
def get_figure(self, figsize=(10, 10), dpi=300):
    """
    Return a pyplot figure
    :param figsize:
    :param dpi:
    :return:
    """
    fig = plt.figure(num=0, figsize=figsize, dpi=dpi)
    fig.clf()
    return fig


def fig_to_rgb_array(self, fig, expand=True):
    """
    Convert figure into a RGB array
    :param fig:         PyPlot Figure
    :param expand:      Flag to expand
    :return:            RGB array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def figure_to_summary(self, fig, summary, place_holder):
    """
    Convert figure into TF summary
    :param fig:             Figure
    :param summary:         Summary to eval
    :param place_holder:    Summary image placeholder
    :return:                Summary
    """
    image = self.fig_to_rgb_array(fig)
    return summary.eval(feed_dict={place_holder: image})


# Converting matrix data into a prettified confusion image
def confusion_matrix_to_image_summary(self, confusion_matrix, summary, place_holder,
                                      list_classes, figsize=(9, 9)):
    """
    Plot confusion matrix and return as TF summary
    :param matrix:          Confusion matrix (N x N)
    :param filename:        Filename
    :param list_classes:    List of classes (N)
    :param figsize:         Pyplot figsize for the confusion image
    :return:                /
    """
    fig = self.get_figure(figsize=(9, 9))
    df = pd.DataFrame(confusion_matrix, index=list_classes, columns=list_classes)
    ax = sns.heatmap(df, annot=True, fmt='.0%')
    # Whatever embellishments you want:
    plt.title('Confusion matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    image_sum = self.figure_to_summary(fig, summary, place_holder)
    return image_sum


# Defining your evaluation operations & Preparing placeholder
# Inspired by Jerod's answer on SO (https://stackoverflow.com/a/42857070/624547)
def add_evaluation_step(self, result_tensor, ground_truth_tensor, num_classes, confusion_matrix_figsize=(9, 9)):
    """
    Sets up the evaluation operations, computing the running accuracy and confusion image
    :param result_tensor:               Output tensor
    :param ground_truth_tensor:         Target class tensor
    :param num_classes:                 Number of classes
    :param confusion_matrix_figsize:    Pyplot figsize for the confusion image
    :return:                            TF operations, summaries and placeholders (see usage below)
    """
    # scope = "evaluation"
    # with tf.name_scope(scope):
    if True:
        # predictions = tf.argmax(result_tensor, 1, name="prediction")

        # Streaming accuracy (lookup and update tensors):
        # accuracy, accuracy_update = tf.metrics.accuracy(ground_truth_tensor, predictions, name='accuracy')
        # Per-batch confusion matrix:
        # batch_confusion = tf.confusion_matrix(ground_truth_tensor, predictions, num_classes=num_classes,
        #                                       name='batch_confusion')

        # Aggregated confusion matrix:
        confusion_matrix = tf.Variable(tf.zeros([num_classes, num_classes], dtype=tf.int32),
                                       name='confusion')
        confusion_update = confusion_matrix.assign(confusion_matrix + batch_confusion)

        # We suppose each batch contains a complete class, to directly normalize by its size:
        evaluate_streaming_metrics_op = tf.group(accuracy_update, confusion_update)

        # Confusion image from matrix (need to extend dims + cast to float so tf.summary.image renormalizes to [0,255]):
        confusion_image = tf.reshape(tf.cast(confusion_update, tf.float32), [1, num_classes, num_classes, 1])

        # Summaries:
        tf.summary.scalar('accuracy', accuracy, collections=[scope])
        summary_op = tf.summary.merge_all(scope)

        # Preparing placeholder for confusion image (so that we can pass the plotted image to it):
        #      (we basically pre-allocate a plot figure and pass its RGB array to a placeholder)
        confusion_image_placeholder = tf.placeholder(tf.uint8,
                                                     fig_to_rgb_array(
                                                         get_figure(figsize=confusion_matrix_figsize)).shape)
        confusion_image_summary = tf.summary.image('confusion_image', confusion_image_placeholder)

    # Isolating all the variables stored by the metric operations:
    running_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    running_vars += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)

    # Initializer op to start/reset running variables
    reset_streaming_metrics_op = tf.variables_initializer(var_list=running_vars)

    return evaluate_streaming_metrics_op, reset_streaming_metrics_op, summary_op, confusion_image_summary, \
        confusion_image_placeholder, confusion_image


# Putting everything together
def evaluate(self, session, model, eval_data_gen):
    """
    Evaluate the model
    :param session:         TF session
    :param eval_data_gen:   Data to evaluate on
    :return:                Evaluation summaries for Tensorboard
    """
    # Resetting streaming vars:
    session.run(reset_streaming_metrics_op)

    # Evaluating running ops over complete eval dataset, e.g.:
    for batch in eval_data_gen:
        feed_dict = {model.inputs: batch}
        session.run(evaluate_streaming_metrics_op, feed_dict=feed_dict)

    # Obtaining the final results:
    summary_str, confusion_results = session.run([summary_op, confusion_image])

    # Converting confusion data into plot into summary:
    confusion_img_str = confusion_matrix_to_image_summary(
        confusion_results[0, :, :, 0], confusion_image_summary, confusion_image_placeholder, classes)
    summary_str += confusion_img_str

    return summary_str  # to be given to a SummaryWriter
