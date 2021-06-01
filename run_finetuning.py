# Finetuning script for YNAT(KLUE-TC) dataset

import collections
import datetime
import json
import os
import random
import re
import time
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,)
from tqdm import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from utils.dataprocessor import YnatProcessor


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# 시간 표시 함수 - hh:mm:ss으로 형태 변경
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, train_dataloader, optimizer, scheduler, device):
    print("Training...")

    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

    train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    return model, optimizer


def validation(model, validation_dataloader, device):
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()
    eval_accuracy, nb_eval_steps = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logit = outputs[0]
        logits = logit.detach().cpu().numpy()

        label_ids = b_labels.to("cpu").numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))


def run():
    # 재현을 위해 랜덤시드 고정
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("%d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU available, using the CPU instead.")

    config = {"batch_size": 4, "num_epochs": 5}

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    processor = YnatProcessor(tokenizer)
    label_list = list(processor.get_labels())

    train_data = processor.get_train_data("data/target_data/ynat-v1")
    validation_data = processor.get_dev_data("data/target_data/ynat-v1")

    train_dataloader = DataLoader(
        train_data, sampler=RandomSampler(train_data), batch_size=config["batch_size"]
    )
    validation_dataloader = DataLoader(
        validation_data,
        sampler=SequentialSampler(validation_data),
        batch_size=config["batch_size"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_list)
    )
    model.to(device)
    model.zero_grad()  # 그래디언트 초기화

    total_steps = len(train_dataloader) * config["num_epochs"]
    print("total steps : ", total_steps)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 에폭만큼 반복
    for epoch in range(config["num_epochs"]):
        print("")
        print(
            "======== Epoch {:} / {:} ========".format(epoch + 1, config["num_epochs"])
        )
        print("Training...")

        model, optimizer = train(model, train_dataloader, optimizer, scheduler, device)
        validation(model, validation_dataloader, device)

        # save checkpoint for each epoch
        torch.save(
            (model.state_dict(), optimizer.state_dict()), f"./model_{epoch+1}.pt"
        )
