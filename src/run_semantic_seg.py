#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional

import evaluate
import transformers
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSemanticSegmentation,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from dotenv import load_dotenv

from custom_segformer import SegFormerWeightnedLoss
from sklearn.model_selection import KFold

""" Finetuning any 🤗 Transformers model supported by AutoModelForSemanticSegmentation for semantic segmentation leveraging the Trainer API."""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

require_version(
    "datasets>=2.0.0",
    "To fix: pip install -r examples/pytorch/semantic-segmentation/requirements.txt",
)


def pad_if_smaller(img, size, fill=0):
    size = (size, size) if isinstance(size, int) else size
    original_width, original_height = img.size
    pad_height = size[1] - original_height if original_height < size[1] else 0
    pad_width = size[0] - original_width if original_width < size[0] else 0
    img = functional.pad(img, (0, 0, pad_width, pad_height), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Identity:
    def __init__(self):
        pass

    def __call__(self, image, target):
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = functional.resize(image, self.size)
        target = functional.resize(
            target, self.size, interpolation=transforms.InterpolationMode.NEAREST
        )
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = functional.resize(image, size)
        target = functional.resize(
            target, size, interpolation=transforms.InterpolationMode.NEAREST
        )
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(image, self.size)
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = functional.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = functional.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ReduceLabels:
    def __call__(self, image, target):
        if not isinstance(target, np.ndarray):
            target = np.array(target).astype(np.uint8)
        # avoid using underflow conversion
        target[target == 0] = 255
        target = target - 1
        target[target == 254] = 255

        target = Image.fromarray(target)
        return image, target


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="segments/sidewalk-semantic",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    reduce_labels: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to reduce all labels by 1 and replace background by 255."
        },
    )

    image_size: Optional[int] = field(
        default=512, metadata={"help": ("Image size to resize/crop for segmentation")}
    )

    n_fold: Optional[int] = field(
        default=0, metadata={"help": ("Folder number 0 to 4")}
    )

    leishmania_weights: Optional[int] = field(
        default=5, metadata={"help": "Leishmania Weights"}
    )

    def __post_init__(self):
        if self.dataset_name is None and (
            self.train_dir is None and self.validation_dir is None
        ):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="nvidia/mit-b0",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_semantic_segmentation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load dataset
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # TODO support datasets from local folders
    # dataset = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir)
    from datasets import Dataset, DatasetDict, Image
    from glob import glob

    # your images can of course have a different extension
    # semantic segmentation maps are typically stored in the png format
    image_paths_train = glob("data/raw_data/*.png")
    label_paths_train = glob("data/label/*")

    def create_dataset(image_paths, label_paths):
        dataset = Dataset.from_dict(
            {"pixel_values": sorted(image_paths), "label": sorted(label_paths)}
        )
        dataset = dataset.cast_column("pixel_values", Image())
        dataset = dataset.cast_column("label", Image())

        return dataset

    # step 1: create Dataset objects
    train_dataset = create_dataset(image_paths_train, label_paths_train)

    dataset = train_dataset.train_test_split(test_size=0.1, seed=42)

    folds = KFold(5, shuffle=True, random_state=42)
    splits = folds.split(dataset["train"])

    print(f'\n[ INFO ] SELECTING THE {data_args.n_fold}')
    train_idxs, val_idxs = list(splits)[data_args.n_fold]

    dataset = DatasetDict(
        {
            "train": dataset["train"].select(train_idxs),
            "validation": dataset["train"].select(val_idxs),
            "test": dataset["test"],
        }
    )

    # Rename column names to standardized names (only "image" and "label" need to be present)
    if "pixel_values" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"pixel_values": "image"})
    if "annotation" in dataset["train"].column_names:
        dataset = dataset.rename_columns({"annotation": "label"})

    label2id = {
        "fundo": 0,
        "leishmania": 1,
        "macrofago_contavel": 2,
        "macrofago_nao_contavel": 3,
    }

    id2label = {v: k for k, v in label2id.items()}

    # Load the mean IoU metric from the datasets package
    metric = evaluate.load("mean_iou")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    @torch.no_grad()
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=None,
            reduce_labels=feature_extractor.do_reduce_labels,
        )
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update(
            {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
        )
        metrics.update(
            {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)}
        )

        return metrics

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.loss_weights = [1, data_args.leishmania_weights, 1, 1]

    
    # AutoModelForSemanticSegmentation
    # SegFormerWeightnedLoss
    model = SegFormerWeightnedLoss.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        size={"height": data_args.image_size, "width": data_args.image_size},
    )

    # Define torchvision transforms to be applied to each image + target.
    # Not that straightforward in torchvision: https://github.com/pytorch/vision/issues/9
    # Currently based on official torchvision references: https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
    if "shortest_edge" in feature_extractor.size:
        # We instead set the target size as (shortest_edge, shortest_edge) to here to ensure all images are batchable.
        size = (
            feature_extractor.size["shortest_edge"],
            feature_extractor.size["shortest_edge"],
        )
    else:
        size = (feature_extractor.size["height"], feature_extractor.size["width"])

    train_transforms = Compose(
        [
            ReduceLabels() if data_args.reduce_labels else Identity(),
            RandomCrop(size=size),
            RandomHorizontalFlip(flip_prob=0.5),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(
                mean=feature_extractor.image_mean, std=feature_extractor.image_std
            ),
        ]
    )
    # Define torchvision transform to be applied to each image.
    # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    val_transforms = Compose(
        [
            ReduceLabels() if data_args.reduce_labels else Identity(),
            Resize(size=size),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(
                mean=feature_extractor.image_mean, std=feature_extractor.image_std
            ),
        ]
    )

    def preprocess_train(example_batch):
        pixel_values = []
        labels = []
        for image, target in zip(example_batch["image"], example_batch["label"]):
            image, target = train_transforms(image.convert("RGB"), target)
            pixel_values.append(image)
            labels.append(target)

        encoding = dict()
        encoding["pixel_values"] = torch.stack(pixel_values)
        encoding["labels"] = torch.stack(labels)

        return encoding

    def preprocess_val(example_batch):
        pixel_values = []
        labels = []
        for image, target in zip(example_batch["image"], example_batch["label"]):
            image, target = val_transforms(image.convert("RGB"), target)
            pixel_values.append(image)
            labels.append(target)

        encoding = dict()
        encoding["pixel_values"] = torch.stack(pixel_values)
        encoding["labels"] = torch.stack(labels)

        return encoding

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                # .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(preprocess_train)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(preprocess_val)

    # ---------------------------TEST---------------------
    # dataset["validation"] = dataset["train"]
    # ---------------------------TEST---------------------

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    load_dotenv()
    main()
