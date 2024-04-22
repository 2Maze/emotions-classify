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

    def draw_confusion_matrix(self, y_pred: Tensor, y_true: Tensor, epoch: int, tensorboard_logger,
                              metric_name="Confusion matrix") -> None:
        figsize = (10, 8)
        confusion_matrix = self.confusion_matrix(y_pred, y_true).cpu()

        df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        fontsize_ticks = 36
        fontsize_ax = 18
        fontsize_title = 24
        fontsize_annotation = 18
        plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        plt.title('Confusion Matrix', fontsize=fontsize_title)

        # Show all ticks and label them with the respective list entries
        font = {'size': 24}
        plt.rc('font', **font)

        hmap = sns.heatmap(
            df_cm, ax=ax, annot=True, square=True,
            fmt=".3g", cmap='Spectral',
            annot_kws={'size': str(fontsize_annotation)},
        )
        plt.ylabel('Real', fontsize=fontsize_ax).set_rotation(90)
        plt.xlabel('Predicted', fontsize=fontsize_ax).set_rotation(0)
        ax.set_xticks(np.arange(len(self.lables)), labels=self.lables)
        ax.set_yticks(np.arange(len(self.lables)), labels=self.lables)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")
        fig.tight_layout()
        plt.close(fig)

        tensorboard_logger.experiment.add_figure(metric_name, fig, epoch)
