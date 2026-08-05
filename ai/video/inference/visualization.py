"""Visualization module for video deepfake detection results.

This module generates visual diagnostic plots for video predictions:
- Probability timeline curves showing confidence over video timestamps.
- Color-coded confidence heatmaps mapping predictions for each sampled frame.
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

import sys
# Configure path references to enable direct execution
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.video.inference.config import InferenceConfig


class VideoInferenceVisualizer:
    """Generates visual analytics plots and timeline reports for model predictions."""

    def __init__(self, config: InferenceConfig):
        """Initializes the VideoInferenceVisualizer.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("video_inference.visualization")
        self.config.create_directories()

    def generate_all_plots(
        self,
        video_name: str,
        frame_timestamps: List[float],
        frame_probs: np.ndarray,
        prediction: str,
        overall_confidence: float
    ) -> None:
        """Generates and saves visual reports to the visualizations directory.

        Args:
            video_name: Name of the processed video.
            frame_timestamps: Timestamp position of each sampled frame in seconds.
            frame_probs: Softmax probabilities of each frame, shape [num_frames, 2].
            prediction: Overall video classification outcome ('Real' or 'Fake').
            overall_confidence: Overall classification confidence score.
        """
        if not self.config.visualization.enable_plots:
            self.logger.info("Visualizations disabled by configuration.")
            return

        # Sanitize video name for filename safety
        safe_name = Path(video_name).stem
        self.logger.info(f"Generating diagnostic plots for video: {safe_name}")

        fake_probs = frame_probs[:, 1] if frame_probs.shape[1] > 1 else frame_probs[:, 0]

        # 1. Plot Probability Timeline Curve
        self._plot_probability_timeline(
            safe_name=safe_name,
            timestamps=frame_timestamps,
            fake_probs=fake_probs,
            prediction=prediction,
            overall_confidence=overall_confidence
        )

        # 2. Plot Frame Confidence Heatmap Grid
        self._plot_confidence_heatmap(
            safe_name=safe_name,
            fake_probs=fake_probs
        )

    def _plot_probability_timeline(
        self,
        safe_name: str,
        timestamps: List[float],
        fake_probs: np.ndarray,
        prediction: str,
        overall_confidence: float
    ) -> None:
        """Generates a line plot showing deepfake probability over video timestamps."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        
        # Plot curve
        plt.plot(timestamps, fake_probs, color="blue", marker="o", linestyle="-", linewidth=2, label="Fake Probability")
        
        # Decision boundary
        threshold = self.config.prediction.confidence_threshold
        plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Decision Threshold ({threshold})")

        # Fill colors representing Real/Fake zones
        plt.fill_between(timestamps, fake_probs, threshold, where=(fake_probs >= threshold), color="red", alpha=0.1, interpolate=True)
        plt.fill_between(timestamps, fake_probs, threshold, where=(fake_probs < threshold), color="green", alpha=0.1, interpolate=True)

        plt.ylim([-0.05, 1.05])
        plt.xlabel("Timeline Position (seconds)")
        plt.ylabel("Fake Probability")
        plt.title(f"Deepfake Probability Timeline: {safe_name}\n(Overall: {prediction} | Conf: {overall_confidence:.2%})")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right")
        
        plt.tight_layout()
        timeline_path = self.config.visualization.output_viz_dir / f"{safe_name}_probability_timeline.png"
        plt.savefig(timeline_path, dpi=150)
        plt.close()
        self.logger.debug(f"Saved timeline plot to: {timeline_path.name}")

    def _plot_confidence_heatmap(
        self,
        safe_name: str,
        fake_probs: np.ndarray
    ) -> None:
        """Generates a 1D grid heatmap of classifications for each frame."""
        import matplotlib.pyplot as plt
        num_frames = len(fake_probs)
        heatmap_data = fake_probs.reshape(1, num_frames)

        plt.figure(figsize=(10, 3.5))
        
        # Colormap mapping Red (Fake) to Green (Real) reversed
        im = plt.imshow(heatmap_data, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
        
        # Add color bar
        cbar = plt.colorbar(im, orientation="horizontal", pad=0.25)
        cbar.set_label("Fake Probability")

        # Set axes labels and ticks
        plt.yticks([])
        plt.xticks(np.arange(num_frames), [f"F_{i+1}" for i in range(num_frames)], rotation=45)
        plt.xlabel("Sampled Frame Order")
        plt.title(f"Frame Classification Confidence Map: {safe_name}")

        # Annotate percentages inside each cell
        thresh = 0.5
        for i in range(num_frames):
            val = fake_probs[i]
            text_color = "white" if (val > 0.85 or val < 0.15) else "black"
            plt.text(
                i, 0, f"{val:.1%}",
                ha="center", va="center",
                color=text_color,
                fontsize=8, fontweight="bold"
            )

        plt.tight_layout()
        heatmap_path = self.config.visualization.output_viz_dir / f"{safe_name}_confidence_heatmap.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        self.logger.debug(f"Saved confidence heatmap plot to: {heatmap_path.name}")
