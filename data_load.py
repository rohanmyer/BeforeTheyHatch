import tensorflow as tf
import numpy as np
import os
import glob
from skimage import io
from PIL import Image

import warnings

warnings.filterwarnings("ignore", module=".*av.*")

import logging

logging.getLogger("libav").setLevel(logging.ERROR)

import utils


class TFSeqDataset(tf.data.Dataset):
    """TensorFlow dataset for sampling sequences of frames from videos"""

    def __new__(
        cls, video_paths, tr_len, img_size, resize_image, augment_data, all_foc_plans
    ):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(None, img_size, img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
            args=(
                video_paths,
                tr_len,
                img_size,
                resize_image,
                augment_data,
                all_foc_plans,
            ),
        )

    @staticmethod
    def _generator(
        video_paths, tr_len, img_size, resize_image, augment_data, all_foc_plans
    ):
        """Generator function to load and process frames from video files"""
        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_nb = utils.getVideoFrameNb(video_path)
            frame_inds = np.arange(frame_nb)

            frame_start = int((gt == -1).sum())
            frame_inds_slice = frame_inds[frame_start : frame_start + tr_len]
            gt_slice = gt[frame_start : frame_start + tr_len]

            frames, gt, video_name, frame_indices = load_frames_and_process(
                frame_inds_slice,
                gt_slice,
                video_name,
                video_path,
                img_size,
                resize_image,
                augment_data,
                all_foc_plans,
            )

            yield frames, gt, video_name, frame_indices


def load_frames_and_process(
    frame_inds,
    gt,
    video_name,
    video,
    img_size,
    resize_image,
    augment_data,
    all_foc_plans,
):
    """Function to load and preprocess frames from disk"""
    all_plans_imgs = []

    frames = tf.concat(all_plans_imgs, axis=0)
    return frames, gt, video_name, frame_inds


def find_frame_ind(path):
    file_name = os.path.splitext(os.path.basename(path))[0]
    frame_ind = int(file_name.split("RUN")[1])
    return frame_ind


# Functions like getGT and utils.getVideoFrameNb should be adapted for TensorFlow usage as needed.
