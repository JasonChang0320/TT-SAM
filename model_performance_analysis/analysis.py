import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

class Precision_Recall_Factory:
    def pga_to_intensity(value):
        pga_threshold = np.log10([0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
        intensity = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
        for i, threshold in enumerate(pga_threshold):
            if value < threshold:
                return intensity[i]
        return intensity[-1]

    def plot_intensity_confusion_matrix(
        intensity_confusion_matrix,
        intensity_score,
        mask_after_sec,
        output_path=None,
    ):
        intensity=["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
        sn.set(rc={"figure.figsize": (8, 8)}, font_scale=1.2)  # for label size
        fig, ax = plt.subplots()
        sn.heatmap(
            intensity_confusion_matrix,
            ax=ax,
            xticklabels=intensity,
            yticklabels=intensity,
            fmt="g",
            annot=True,
            annot_kws={"size": 16},
            cmap="Reds",
            cbar=True,
            cbar_kws={"label": "number of traces"},
        )  # font size
        for i in range(len(intensity)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="gray", lw=2))
        ax.set_xlabel("Predicted intensity", fontsize=18)
        ax.set_ylabel("Actual intensity", fontsize=18)
        ax.set_title(
            f"{mask_after_sec} sec intensity confusion matrix, intensity score: {np.round(intensity_score,3)}"
        )
        if output_path:
            fig.savefig(
                f"{output_path}/{mask_after_sec} sec intensity confusion matrix.png",
                dpi=300,
            )
        return fig, ax

    def plot_score_curve(
        performance_score,
        fig,
        ax,
        score_type,
        score_curve_threshold,
        mask_after_sec,
        output_path=None,
    ):
        ax.plot(
            100 * (10**score_curve_threshold),
            performance_score[f"{score_type}"],
            label=f"{mask_after_sec} sec",
        )
        ax.set_xlabel(r"PGA threshold (${cm/s^2}$)", fontsize=15)
        ax.set_ylabel("score", fontsize=15)
        ax.set_title(f"{score_type} curve", fontsize=22)
        ax.set_ylim(0, 1.1)
        ax.legend()
        if output_path:
            fig.savefig(f"{output_path}/{score_type}_curve.png", dpi=300)
        return fig, ax
