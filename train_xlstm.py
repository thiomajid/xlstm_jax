#!/usr/bin/env python3
import json
import logging
from typing import cast

import hydra
import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
import orbax.checkpoint as orbax

from src import xLSTMLMModel
from src._trainer.arguments import CustomArgs
from src._trainer.data import get_dataset
from src.utils import parse_xlstm_config_dict


def loss_fn(model: xLSTMLMModel, batch: tuple[jnp.ndarray, ...]):
    """Compute the loss for a batch of data."""
    input_ids, labels = batch
    logits = model(input_ids)

    # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
    shifted_logits = rearrange(logits[..., :-1, :], "b s v -> (b s) v")

    # shape: [batch, seq] -> [batch * (seq-1)]
    shifted_labels = rearrange(labels[..., 1:], "b s -> (b s)")

    # Compute cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shifted_logits, labels=shifted_labels
    ).mean()

    return loss, logits


@nnx.jit
def train_step(
    model: xLSTMLMModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: tuple[jnp.ndarray, ...],
):
    """Perform a single training step."""

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (loss, logits), grads = grad_fn(model, batch)

    perplexity = jnp.exp(loss)
    metrics.update(loss=loss, perplexity=perplexity)

    # jax.debug.print("Gradients shape: {grads}", grads=grads)
    optimizer.update(grads)

    return loss


@nnx.jit
def apply_gradients(
    optimizer: nnx.Optimizer,
    grads: jax.Array,
):
    """Apply gradients to the model."""
    optimizer.update(grads=grads)


@nnx.jit
def eval_step(
    model: xLSTMLMModel,
    metrics: nnx.MultiMetric,
    batch: tuple[jnp.ndarray, ...],
):
    """Perform a single evaluation step."""
    loss, logits = loss_fn(model, batch)
    perplexity = jnp.exp(loss)
    metrics.update(loss=loss, perplexity=perplexity)


@hydra.main(config_path="./configs", config_name="train_config")
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    parser = HfArgumentParser(CustomArgs)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]
    args = cast(CustomArgs, args)

    logger.info("Loading tokenizer...")
    # Load teacher model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")

    config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    config_dict["vocab_size"] = tokenizer.vocab_size
    print(config_dict)
    config = parse_xlstm_config_dict(config_dict)
    config.pad_token_id = tokenizer.pad_token_id

    # Model instance
    logger.info("Creating xLSTM model...")
    rngs = nnx.Rngs(args.seed, params=jax.random.key(args.seed))
    model = xLSTMLMModel(config, rngs=rngs, dtype=jnp.float32)

    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_dataset = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        features=args.features,
        max_seq_length=config.context_length,
        tokenizer=tokenizer,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset.set_format("numpy", columns=["input_ids", "attention_mask", "length"])

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_dataset = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        features=args.features,
        max_seq_length=config.context_length,
        tokenizer=tokenizer,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    eval_dataset.set_format("numpy", columns=["input_ids", "attention_mask", "length"])

    logger.info("Initializing trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="np",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        drop_last=False,
    )

    # setup the training loop
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=args.learning_rate,
                # b1=args.adam_beta1,
                # b2=args.adam_beta2,
                # weight_decay=args.weight_decay,
            ),
        ),
    )

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
    )

    history = {
        "train_loss": [],
        "train_perplexity": [],
        "eval_loss": [],
        "eval_perplexity": [],
    }

    # checkpoint manager
    checkpoint_manager = orbax.CheckpointManager(
        checkpoint=orbax.Checkpoint(),
        options=orbax.CheckpointOptions(
            keep=5,
            keep_every_n_epochs=1,
            keep_every_n_steps=1000,
        ),
    )

    # Start training with progress bar being updated and gradient accumulation
    # if needed and descriptive messages
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_steps = args.num_train_epochs * steps_per_epoch
    global_step = 0

    logger.info("Starting training loop...")
    with tqdm(total=max_steps, desc="Training") as pbar:
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_loader):
                # Update the model
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"])

                _batch = (input_ids, labels)
                _ = train_step(model, optimizer, metrics, _batch)
                global_step += 1

                # Update the progress bar
                pbar.update(1)

                if global_step % args.logging_steps == 0:
                    # Log the metrics
                    for metric, value in metrics.compute().items():
                        history[f"train_{metric}"].append(value.item())

                        # update progress bar description
                        pbar.set_postfix({metric: value.item()})

                    pbar.refresh()
                    metrics.reset()

            # Evaluate the model after each epoch
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"])
                _batch = (input_ids, labels)
                eval_step(model=model, batch=_batch, metrics=metrics)

            for metric, value in metrics.compute().items():
                history[f"eval_{metric}"].append(value.item())

                # update progress bar description
                pbar.set_postfix({f"eval_{metric}": value.item()})

            pbar.refresh()
            metrics.reset()

    # save metrics to a json file
    with open("./train_history.json", "w") as f:
        json.dump(history, f, indent=4)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
