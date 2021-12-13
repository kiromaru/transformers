import logging
import math
import os
import re
import shutil
import warnings
import csv
import random
import functools
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, default_data_collator
from .file_utils import is_apex_available, is_torch_tpu_available
from .modeling_utils import PreTrainedModel
from .tokenization_utils import PreTrainedTokenizer
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    is_wandb_available,
    set_seed,
)
from .training_args import TrainingArguments
from .data.datasets.glue import GlueDataset


if is_apex_available():
    from apex import amp


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        print("Could not import tensorboard from either torch.utils.tensorboard or tensorboardX")
        raise EnvironmentError
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

def compare_replacement_tokens(t1, t2):
    len1 = len(t1)
    len2 = len(t2)

    max_len = max(len1, len2)

    for idx in range(max_len):
        if idx >= len1 and idx < len2:
            return -1
        if idx < len1 and idx >= len2:
            return 1
        if t1[idx] < t2[idx]:
            return -1
        if t1[idx] > t2[idx]:
            return 1;
    return 0

class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TrainingArguments`):
            The arguments to tweak training.
        data_collator (:obj:`DataCollator`, `optional`, defaults to :func:`~transformers.default_data_collator`):
            The function to use to from a batch from a list of elements of :obj:`train_dataset` or
            :obj:`eval_dataset`.
        train_dataset (:obj:`Dataset`, `optional`):
            The dataset to use for training.
        eval_dataset (:obj:`Dataset`, `optional`):
            The dataset to use for evaluation.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and predictions, only returns the loss.
        tb_writer (:obj:`SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None
    prediction_model: Optional[PreTrainedModel] = None
    prediction_tokenizer: Optional[PreTrainedTokenizer]
    prediction_optimizer: Optional[torch.optim.Optimizer]
    loss_writer: Optional["SummaryWriter"] = None
    word_replacement_tokens: Optional[List[List[int]]]

    prediction_layers = [
        'cls.predictions.bias',
        'cls.predictions.transform.dense.weight',
        'cls.predictions.transform.dense.bias',
        'cls.predictions.transform.LayerNorm.weight',
        'cls.predictions.transform.LayerNorm.bias',
        'cls.predictions.decoder.weight',
        'cls.predictions.decoder.bias'
    ]
    classifier_layers = [
        'classifier.weight',
        'classifier.bias',
        'classification_head.dense.weight',
        'classification_head.dense.bias',
        'classification_head.out_proj.weight',
        'classification_head.out_proj.bias'
    ]

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        data_dir: Optional[str] = None,
        prediction_model: Optional[PreTrainedModel] = None,
        prediction_tokenizer: Optional[PreTrainedTokenizer] = None,
        switched_model: Optional[PreTrainedModel] = None,
        word_replacement_list: Optional[List[str]] = None,
    ):
        self.model = model.to(args.device)
        self.args = args
        self.data_collator = data_collator if data_collator is not None else default_data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.data_dir = data_dir
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                    "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                    + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )
        # Setup log file if reporting loss
        if self.args.log_loss:
            loss_log_filename = os.path.join(self.args.output_dir, "loss.csv")
            loss_log_file = open(loss_log_filename, 'w', encoding='utf-8', newline='')
            self.tsv_loss_log = csv.writer(loss_log_file, delimiter='\t')

        if prediction_model is not None:
            self.prediction_model = prediction_model.to(args.device)
            self.prediction_tokenizer = prediction_tokenizer
            if self.prediction_tokenizer is not None:
                self.special_tokens = [
                    self.prediction_tokenizer.cls_token_id,
                    self.prediction_tokenizer.sep_token_id,
                    self.prediction_tokenizer.mask_token_id,
                    self.prediction_tokenizer.eos_token_id,
                    self.prediction_tokenizer.bos_token_id,
                    self.prediction_tokenizer.pad_token_id
                ]

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.prediction_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.prediction_model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.prediction_optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        if switched_model is not None:
            self.switched_model = switched_model.to(args.device)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.switched_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.switched_model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.switched_optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        if is_tensorboard_available():
            self.loss_writer = SummaryWriter(log_dir=self.args.logging_dir)

        self.word_replacement_tokens = []
        self.first_word_tokens = []
        self.first_word_token_set = set()

        if word_replacement_list is not None:
            for word in word_replacement_list:
                if len(word) > 0:
                    encoded_word = self.prediction_tokenizer.encode(word)
                    encoded_word_no_st = [x for x in encoded_word if x not in self.special_tokens ]
                    self.word_replacement_tokens.append(encoded_word_no_st)

            self.word_replacement_tokens = sorted(self.word_replacement_tokens, key=functools.cmp_to_key(compare_replacement_tokens), reverse=True)

            for word_tokens in self.word_replacement_tokens:
                self.first_word_tokens.append(word_tokens[0])
                if word_tokens[0] not in self.first_word_token_set:
                    self.first_word_token_set.add(word_tokens[0])

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Args:
            test_dataset (obj:`Dataset`): The test dataset to use.
        """
        # We use the same batch_size as for eval.
        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        if self.is_world_master():
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(
                    self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None, train_dataset_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # Get the word-prediction loss for the batch and pass it to _training_step. This method is the
                # one that calls loss.backward()
                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_master():
                wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)

    def _copy_weights_to_prediction_model(self, source_model: nn.Module):
        # Copy current weights to prediction model
        prediction_state = self.prediction_model.state_dict()
        classification_state = source_model.state_dict()
        classification_keys = classification_state.keys()

        for model_key in classification_keys:
            if not model_key in self.classifier_layers:
                prediction_state[model_key] = classification_state[model_key]
        self.prediction_model.load_state_dict(prediction_state)

    def _replace_words_with_masks_targeted(self, input: torch.Tensor) -> bool:
        made_replacement = False
        for token_idx in range(len(input)):
            token = input[token_idx]
            if token not in self.special_tokens and token.item() in self.first_word_token_set:
                word_idxs = []
                for fwt_idx in range(len(self.first_word_tokens)):
                    if self.first_word_tokens[fwt_idx] == token.item():
                        word_idxs.append(fwt_idx)

                for word_idx in word_idxs:
                    if 1 == len(self.word_replacement_tokens[word_idx]):
                        input[token_idx] = self.prediction_tokenizer.mask_token_id
                        made_replacement = True

                        # Don't process any more word_idxs
                        break
                    else:
                        # Need to replace the full length of the word
                        full_word_found = True
                        word_len = len(self.word_replacement_tokens[word_idx])
                        for list_word_idx in range(word_len):
                            if input[token_idx + list_word_idx] != self.word_replacement_tokens[word_idx][list_word_idx]:
                                full_word_found = False
                                break

                        if full_word_found:
                            # Shift sentence to the left after inserting mask
                            input[token_idx] = self.prediction_tokenizer.mask_token_id
                            for sentence_idx in range(token_idx + word_len, len(input)):
                                input[sentence_idx - word_len + 1] = input[sentence_idx]
                            # Fill up the end
                            for sentence_idx in range(len(input) - word_len + 1, len(input)):
                                input[sentence_idx] = self.prediction_tokenizer.pad_token_id

                            made_replacement = True
                            # Don't process any more word_idxs
                            break

        return made_replacement

    def _replace_words_with_masks(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        replace_percentage = self.args.word_replacement_pct
        for sentence_idx in range(len(inputs['input_ids'])):
            if self._replace_words_with_masks_targeted(inputs['input_ids'][sentence_idx]):
                continue

            sep_idxs = torch.where(inputs['input_ids'][sentence_idx] == self.prediction_tokenizer.sep_token_id)
            max_sep_idx = torch.argmax(sep_idxs[0])
            last_separator_idx = sep_idxs[0][max_sep_idx].item()
            words_to_replace = int(round(last_separator_idx * replace_percentage))
            while words_to_replace > 0:
                replace_idx = random.randrange(last_separator_idx)
                if inputs['input_ids'][sentence_idx][replace_idx] not in self.special_tokens:
                    inputs['input_ids'][sentence_idx][replace_idx] = self.prediction_tokenizer.mask_token_id
                    words_to_replace -= 1

    def _replace_predicted_words(self, inputs: Dict[str, Union[torch.Tensor, Any]], token_logits):
        for sentence_idx in range(len(inputs['input_ids'])):
            mask_token_index = torch.where(inputs['input_ids'][sentence_idx] == self.prediction_tokenizer.mask_token_id)
            if len(mask_token_index[0]) > 0:
                mask_token_logits = token_logits[sentence_idx, mask_token_index[0], :]
                predicted_indexes = torch.argmax(mask_token_logits, dim=1)
                for predicted_index in range(len(mask_token_index[0])):
                    if predicted_indexes[predicted_index] not in self.special_tokens:
                        inputs['input_ids'][sentence_idx][mask_token_index[0][predicted_index]] = predicted_indexes[predicted_index]

    def _switch_input_sentences(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        for sentence_idx in range(len(inputs['input_ids'])):
            sep_idxs = torch.where(inputs['input_ids'][sentence_idx] == self.prediction_tokenizer.sep_token_id)

            if (len(sep_idxs[0]) == 2):
                # BERT has 2 separators
                first_sentence_start = 1
                first_sentence_end = sep_idxs[0][0]
                second_sentence_start = sep_idxs[0][0] + 1
                second_sentence_end = sep_idxs[0][1]
                new_separators = [ sep_idxs[0][1] - sep_idxs[0][0] ]
            else:
                # BART has 3 separators
                first_sentence_start = 1
                first_sentence_end = sep_idxs[0][0]
                second_sentence_start = sep_idxs[0][1] + 1
                second_sentence_end = sep_idxs[0][2]
                new_separators = [ sep_idxs[0][2] - sep_idxs[0][1], sep_idxs[0][2] - sep_idxs[0][1] + 1]

            # Save first sentence
            first_sentence = inputs['input_ids'][sentence_idx][first_sentence_start : first_sentence_end].clone()
            second_sentence = inputs['input_ids'][sentence_idx][second_sentence_start : second_sentence_end].clone()

            # Overwrite input with switched output
            for i in range(len(second_sentence)):
                inputs['input_ids'][sentence_idx][i + 1] = second_sentence[i]

            # Separator(s)
            for sep_idx in new_separators:
                inputs['input_ids'][sentence_idx][sep_idx] = self.prediction_tokenizer.sep_token_id

            for i in range(len(first_sentence)):
                inputs['input_ids'][sentence_idx][i + new_separators[-1] + 1] = first_sentence[i]

    def _word_prediction(self, inputs: Dict[str, Union[torch.Tensor, Any]], replace_words: bool) -> float:
        if self.prediction_model is None:
            return 0.0

        self.prediction_model.eval()

        # Original input is the labels
        self.prediction_labels = inputs['input_ids'].clone()

        #self._copy_weights_to_prediction_model(source_model);
        self._replace_words_with_masks(inputs)
        self.masked_input = inputs['input_ids'].clone()

        # Perform word prediction
        prediction_output = self.prediction_model(self.masked_input, labels=self.prediction_labels)
        pre_loss = prediction_output[0]
        token_logits = prediction_output[1]
        pre_loss = pre_loss / self.args.word_prediction_loss_atenuator
        pre_loss = pre_loss * self.args.training_w1

        if replace_words:
            # Replace predicted words in input
            self._replace_predicted_words(inputs, token_logits)

        return pre_loss.item()

    def _word_prediction_training(self, finetuning_loss: float, switch_input_loss: float) -> float:
        if self.prediction_model is None:
            return 0.0

        self.prediction_model.train()

        # We should already have labels and masked input
        # Perform word prediction
        prediction_output = self.prediction_model(self.masked_input, labels=self.prediction_labels)
        pre_loss = prediction_output[0]
        token_logits = prediction_output[1]
        pre_loss = pre_loss / self.args.word_prediction_loss_atenuator

        # Optimization
        pre_loss = pre_loss * self.args.training_w1
        if self.loss_writer:
            self.loss_writer.add_scalar("TrainingLoss/prediction", pre_loss.item())

        # Combined loss
        pre_loss = pre_loss + finetuning_loss + switch_input_loss
        pre_loss.backward()
        self.prediction_optimizer.step()
        self.prediction_model.zero_grad()

        return pre_loss.item()

    def _switch_input_sentences_loss(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> (float, Dict[str, Union[torch.Tensor, Any]]):
        switched_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                switched_inputs[k] = v.clone()

        self._switch_input_sentences(switched_inputs)
        self.switched_model.eval()
        with torch.no_grad():
            # Calculate loss from sentences in switched order
            switch_outputs = self.switched_model(**switched_inputs)
            switch_loss = switch_outputs[0].item()
            switch_loss = self.args.training_w2 * switch_loss

        return (switch_loss, switched_inputs)

    def _switched_input_training(self, inputs: Dict[str, Union[torch.Tensor, Any]], finetuning_loss: float, word_prediction_loss: float) -> float:
        if self.switched_model is None:
            return 0.0

        self.switched_model.train()

        # Input should already contain switched sentences
        switch_output = self.switched_model(**inputs)
        switch_loss = switch_output[0]

        # Optimization
        switch_loss = switch_loss * self.args.training_w2
        if self.loss_writer:
            self.loss_writer.add_scalar("TrainingLoss/switched_sentences", switch_loss.item())

        # Combined loss
        switch_loss = switch_loss + finetuning_loss + word_prediction_loss
        switch_loss.backward()
        self.switched_optimizer.step()
        self.switched_model.zero_grad()

        return switch_loss.item()

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        model.train()

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        pre_loss = 0
        switch_loss = 0

        if self.args.training_w2 != 0.0:
            # Switched input
            switch_loss, switched_inputs = self._switch_input_sentences_loss(inputs)

        if self.args.training_w1 != 0.0:
            # Word prediction
            pre_loss = self._word_prediction(inputs, True)

        fine_tuning_w = 1.0 - self.args.training_w1 - self.args.training_w2

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss = fine_tuning_w * loss

        if self.loss_writer:
            self.loss_writer.add_scalar("TrainingLoss/finetuning", loss.mean().item())

        # Combined loss
        finetuning_loss = loss.mean().item()
        loss = loss + pre_loss + switch_loss

        if self.loss_writer:
            self.loss_writer.add_scalar("TrainingLoss/combined", loss.mean().item())

        if self.args.log_loss:
            logrow = [ self.args.training_w, pre_loss, switch_loss, finetuning_loss, loss.mean().item() ]
            self.tsv_loss_log.writerow(logrow)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.args.training_w1 != 0.0:
            # Now do the actual word prediction training
            self._word_prediction_training(finetuning_loss, switch_loss)

        if self.args.training_w2 != 0.0:
            # Now do the actual switch sentence training
            self._switched_input_training(switched_inputs, finetuning_loss, pre_loss)

        return loss.item()

    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on.
        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            past = None

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            switched_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
                    switched_inputs[k] = inputs[k].clone()
            if self.args.past_index >= 0:
                inputs["mems"] = past

            classification_loss = 0.0
            switch_loss = 0.0
            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    classification_loss = step_eval_loss.mean().item()
                    eval_losses += [classification_loss]
                else:
                    logits = outputs[0]
                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

                if self.args.training_w2 != 0.0 and has_labels and self.loss_writer and self.prediction_model is not None:
                    self._switch_input_sentences(switched_inputs)
                    switch_outputs = model(**switched_inputs)
                    switch_loss = switch_outputs[0].mean().item()

                    self.loss_writer.add_scalar("EvalLoss/switched_sentences", switch_loss)

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

            pre_loss = 0
            if self.args.training_w1 != 0.0 and self.loss_writer:
                # Perform word prediction
                pre_loss = self._word_prediction(inputs, False)
                self.loss_writer.add_scalar("EvalLoss/prediction", pre_loss)

            if self.loss_writer:
                self.loss_writer.add_scalar("EvalLoss/finetuning", classification_loss)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output
