# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name)
        wandb.config.update({"log_dir": log_dir})

        # Video tracking
        self.saved_video_files: dict[str, dict] = {}

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        wandb.config.update({"runner_cfg": train_cfg})
        wandb.config.update({"policy_cfg": train_cfg["policy"]})
        wandb.config.update({"alg_cfg": train_cfg["algorithm"]})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))

    def update_video_files(self, log_name: str, fps: int) -> None:
        """Check for new video files and upload them to wandb."""
        # Check if there are new video files
        log_dir = pathlib.Path(self.log_dir)
        video_files = list(log_dir.rglob("*.mp4"))
        for video_file in video_files:
            file_size_kb = os.stat(str(video_file)).st_size / 1024
            # If it is new file
            if str(video_file) not in self.saved_video_files:
                self.saved_video_files[str(video_file)] = {"size": file_size_kb, "added": False, "count": 0}
            else:
                # Only upload if the file size is not changing anymore to avoid uploading non-ready video.
                video_info = self.saved_video_files[str(video_file)]
                if video_info["added"] is False and video_info["size"] == file_size_kb and file_size_kb > 100:
                    if video_info["count"] > 10:
                        print(f"[Wandb] Uploading {os.path.basename(str(video_file))}.")
                        wandb.log({log_name: wandb.Video(str(video_file), fps=fps)})
                        self.saved_video_files[str(video_file)]["added"] = True
                    else:
                        video_info["count"] += 1
                else:
                    self.saved_video_files[str(video_file)]["size"] = file_size_kb
                    video_info["count"] = 0
