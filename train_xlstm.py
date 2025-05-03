import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import cast

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import nnx
from huggingface_hub import create_repo, repo_exists, upload_folder
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)

from xlstm_jax import xLSTMLMModel
from xlstm_jax._trainer.arguments import CustomArgs
from xlstm_jax._trainer.data import get_dataset
from xlstm_jax.utils import parse_xlstm_config_dict, str2dtype


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


@hydra.main(config_path="./configs", config_name="train_config", version_base="1.1")
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
    dtype_str = cfg["dtype"]
    logger.info(f"Creating xLSTM model with dtype={dtype_str}...")
    dtype = str2dtype(dtype_str)
    rngs = nnx.Rngs(args.seed)
    model = xLSTMLMModel(config, rngs=rngs, dtype=dtype)

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
            # optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=args.learning_rate,
                b1=args.adam_beta1,
                b2=args.adam_beta2,
                weight_decay=args.weight_decay,
            ),
        ),
    )

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
    )

    history: dict[str, list[dict[str, float]]] = {
        "train_loss": [],
        "train_perplexity": [],
        "eval_loss": [],
        "eval_perplexity": [],
    }

    # checkpoint manager

    ckpt_dir = ocp.test_utils.erase_and_create_empty(Path(args.logging_dir).absolute())
    checkpointer = ocp.PyTreeCheckpointer()
    CKPT_PREFIX = "state"

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
                        history[f"train_{metric}"].append(
                            {
                                "step": global_step,
                                "value": value.item(),
                            }
                        )

                    # update progress bar description
                    last_loss = history["train_loss"][-1]["value"]
                    last_ppl = history["train_perplexity"][-1]["value"]
                    pbar.set_postfix(
                        {
                            "train_loss": last_loss,
                            "train_perplexity": last_ppl,
                        }
                    )

                    pbar.refresh()
                    metrics.reset()

                if global_step % args.save_steps == 0:
                    # Save the model checkpoint
                    logger.info(f"Saving checkpoint at step {global_step}...")
                    state_dir = ckpt_dir / f"{CKPT_PREFIX}-{global_step}"
                    # state = nnx.to_pure_dict(nnx.state(model))
                    state = nnx.state(model)
                    checkpointer.save(state_dir, state)  # Save filtered state

            # Evaluate the model after each epoch
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"])
                _batch = (input_ids, labels)
                eval_step(model=model, batch=_batch, metrics=metrics)

            for metric, value in metrics.compute().items():
                history[f"eval_{metric}"].append(
                    {
                        "step": global_step,
                        "value": value.item(),
                    }
                )

            # update progress bar description
            last_loss = history["eval_loss"][-1]["value"]
            last_ppl = history["eval_perplexity"][-1]["value"]
            pbar.set_postfix(
                {
                    "eval_loss": last_loss,
                    "eval_perplexity": last_ppl,
                }
            )

            pbar.refresh()
            metrics.reset()

    logger.info("Training completed.")

    logger.info("Saving final model...")
    # save metrics to a json file
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # save the model config
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(
            asdict(config),
            f,
            indent=4,
        )

    # put the checkpoint with the best train_perplexity in artifacts_dir
    train_ppl = history["train_perplexity"]

    # sort the train_ppl by value in ascending order
    sorted_train_ppl = sorted(
        train_ppl,
        key=lambda x: x["value"],
    )

    best_ckpt = sorted_train_ppl[0]
    step = int(best_ckpt["step"])
    ppl_value = best_ckpt["value"]
    logger.info(f"Best checkpoint: {step} with train_perplexity: {ppl_value}")

    # copy the checkpoint to artifacts_dir
    best_ckpt_dir_path = ckpt_dir / f"{CKPT_PREFIX}-{step}"
    best_ckpt_dir_path = best_ckpt_dir_path.resolve()
    shutil.copytree(
        best_ckpt_dir_path,
        artifacts_dir / CKPT_PREFIX,
        dirs_exist_ok=True,
    )

    # save the tokenizer
    tokenizer.save_pretrained(artifacts_dir)

    if args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub...")
        if not repo_exists(args.hub_model_id, token=args.hub_token):
            logger.info(
                f"Creating repository {args.hub_model_id} on Hugging Face Hub..."
            )
            create_repo(
                repo_id=args.hub_model_id,
                token=args.hub_token,
                private=args.hub_private_repo,
                exist_ok=True,
            )

        # Upload the model to the hub
        logger.info(f"Uploading model to {args.hub_model_id}...")
        upload_folder(
            repo_id=args.hub_model_id,
            folder_path=artifacts_dir,
            token=args.hub_token,
            commit_message="Training completed",
        )


if __name__ == "__main__":
    main()
