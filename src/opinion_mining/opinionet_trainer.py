from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .baseline import Candidate
from .data import ReviewExample
from .neural import (
    NeuralConfig,
    _ReviewDataset,
    _collate,
    _default_collate_wrapper,
    _move_batch,
    _pointer_batch_loss,
    choose_device,
    set_seed,
)
from .opinionet_decoder import decode_opinionet_candidates
from .resource_guard import ResourceGuard


class OpinionetModel(nn.Module):
    def __init__(self, encoder: PreTrainedModel, trainable_layers: int, relation_classes: int = 39):
        super().__init__()
        from .opinionet_head import OpinionetHead

        self.encoder = encoder
        self.head = OpinionetHead(int(encoder.config.hidden_size), relation_classes=relation_classes)
        self._configure_trainable_layers(trainable_layers)

    def _configure_trainable_layers(self, trainable_layers: int) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        else:
            for layer in list(layers)[-max(0, trainable_layers) :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        kwargs: dict[str, Tensor] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        return self.encoder(**kwargs).last_hidden_state

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden = self.encode(batch)
        outputs = self.head(hidden)
        outputs["hidden"] = hidden
        return outputs


class OpinionetTrainer:
    def __init__(self, config: NeuralConfig, *, guard: ResourceGuard | None = None):
        self.config = config
        self.guard = guard or ResourceGuard()
        self.device = choose_device(config.device)
        set_seed(config.seed)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def _build_model(self) -> OpinionetModel:
        encoder = AutoModel.from_pretrained(self.config.model_name)
        return OpinionetModel(encoder, self.config.trainable_layers)

    def _predict(self, model: OpinionetModel, rows: list[ReviewExample]) -> dict[int, list[Candidate]]:
        if not rows:
            return {}
        model.eval()
        dataset = _ReviewDataset(
            rows,
            self.tokenizer,
            self.config.max_length,
            use_official_offsets=self.config.use_official_offsets,
            use_implicit_opinion_sentinel=self.config.use_implicit_opinion_sentinel,
            preserve_multi_relation=self.config.preserve_multi_relation,
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.config.batch_size * 2),
            shuffle=False,
            num_workers=0,
            collate_fn=_default_collate_wrapper,
        )
        output: dict[int, list[Candidate]] = {}
        offset = 0
        with torch.no_grad():
            for raw_batch in loader:
                batch = _move_batch(raw_batch, self.device)
                raw_outputs = model(batch)
                cpu_outputs = {
                    key: value.detach().cpu()
                    for key, value in raw_outputs.items()
                    if isinstance(value, Tensor)
                }
                size = len(raw_batch["pairs"])
                for local_index, feature in enumerate(dataset.features[offset : offset + size]):
                    row = rows[offset + local_index]
                    output[row.id] = decode_opinionet_candidates(
                        row,
                        feature,
                        cpu_outputs,
                        row_index=local_index,
                        max_span_length=self.config.max_span_length,
                        span_top_k=self.config.span_top_k,
                        beam_top_k=self.config.beam_top_k,
                        relation_top_k=self.config.relation_top_k,
                    )
                offset += size
        return output

    def train_fold(
        self,
        train_examples: Iterable[ReviewExample],
        valid_examples: Iterable[ReviewExample],
        test_examples: Iterable[ReviewExample] | None = None,
        *,
        output_dir: str | Path | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        train_rows = list(train_examples)
        valid_rows = list(valid_examples)
        test_rows = list(test_examples or [])
        model = self._build_model().to(self.device)
        train_dataset = _ReviewDataset(
            train_rows,
            self.tokenizer,
            self.config.max_length,
            use_official_offsets=self.config.use_official_offsets,
            use_implicit_opinion_sentinel=self.config.use_implicit_opinion_sentinel,
            preserve_multi_relation=self.config.preserve_multi_relation,
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
        head_parameters = [parameter for parameter in parameters if all(parameter is not candidate for candidate in encoder_parameters)]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.config.learning_rate},
                {"params": head_parameters, "lr": self.config.head_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, Any]] = []
        best_f1 = -1.0
        best_state: dict[str, Tensor] | None = None
        batch_size = self.config.batch_size
        for epoch in range(1, self.config.epochs + 1):
            recommendation = self.guard.recommend(self.config)
            batch_size = int(recommendation.get("batch_size", batch_size))
            if recommendation.get("max_length") != self.config.max_length:
                self.config.max_length = int(recommendation["max_length"])
                train_dataset = _ReviewDataset(
                    train_rows,
                    self.tokenizer,
                    self.config.max_length,
                    use_official_offsets=self.config.use_official_offsets,
                    use_implicit_opinion_sentinel=self.config.use_implicit_opinion_sentinel,
                    preserve_multi_relation=self.config.preserve_multi_relation,
                )
            loader = DataLoader(train_dataset, batch_size=max(1, batch_size), shuffle=True, num_workers=0, collate_fn=_default_collate_wrapper)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = {"loss": 0.0, "span_loss": 0.0, "relation_loss": 0.0}
            steps = 0
            for step, raw_batch in enumerate(loader, start=1):
                batch = _move_batch(raw_batch, self.device)
                outputs = model(batch)
                loss, parts = _pointer_batch_loss(model, outputs, batch)
                (loss / max(1, self.config.gradient_accumulation)).backward()
                if step % max(1, self.config.gradient_accumulation) == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for key, value in parts.items():
                    running[key] += value
                steps += 1
            model.eval()
            valid_candidates = self._predict(model, valid_rows)
            from .pipeline import threshold_predictions
            from .metrics import strict_f1

            valid_gold = {row.id: row.labels for row in valid_rows}
            valid_predictions = threshold_predictions(valid_candidates, 0.12, 0.10)
            score = strict_f1(valid_gold, valid_predictions)
            record = {
                "epoch": epoch,
                "step": steps,
                **{key: value / max(1, steps) for key, value in running.items()},
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "resource": asdict(self.guard.snapshot()),
            }
            history.append(record)
            if on_update:
                on_update(record)
            if score.f1 > best_f1:
                best_f1 = score.f1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            self.guard.release_cache()
        if best_state is not None:
            model.load_state_dict(best_state)
        valid_candidates = self._predict(model, valid_rows)
        test_candidates = self._predict(model, test_rows)
        checkpoint = None
        if output_dir is not None:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = directory / "model.pt"
            torch.save({"state_dict": model.state_dict(), "config": asdict(self.config)}, checkpoint_path)
            checkpoint = str(checkpoint_path)
            (directory / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        self.guard.release_cache()
        return {"candidates": valid_candidates, "test_candidates": test_candidates, "history": history, "checkpoint": checkpoint, "config": asdict(self.config)}
