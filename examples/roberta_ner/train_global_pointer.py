# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from tqdm import tqdm
import oneflow as torch
from oneflow.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
from flagai.data.collate_utils import bert_sequence_label_gp_collate_fn
from flagai.model.predictor.predictor import Predictor
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = "./data/china-people-daily-ner-corpus/example.train"
valid_path = './data/china-people-daily-ner-corpus/example.dev'
test_path = './data/china-people-daily-ner-corpus/example.test'

task_name = "sequence-labeling-gp"
model_dir = "./state_dict/"

target = ["LOC", "ORG", "PER"]
maxlen = 256

trainer = Trainer(env_type="pytorch",
                  experiment_name="roberta_ner_global_pointer",
                  batch_size=16,
                  gradient_accumulation_steps=1,
                  lr=2e-5,
                  weight_decay=1e-3,
                  epochs=10,
                  log_interval=100,
                  eval_interval=1000,
                  load_dir=None,
                  pytorch_device=device,
                  save_dir="checkpoints_ner_gp",
                  save_interval=1)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                elif flag[0] == 'I':
                    d[-1][1] = i

            D.append(d)
    return D


train_data = load_data(train_path)
val_data = load_data(valid_path)
test_data = load_data(test_path)

print(f"trian_data is {len(train_data)}")
print(f"val_data is {len(val_data)}")
print(f"test_data is {len(test_data)}")
print(f"target is {target}")

auto_loader = AutoLoader(task_name,
                         model_name="RoBERTa-base-ch",
                         model_dir=model_dir,
                         class_num=len(target),
                         inner_dim=64)

model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


class NERDataset(Dataset):

    def __init__(self, data):
        super(NERDataset, self).__init__()
        self.data = data

    def __getitem__(self, i):
        data = self.data[i]
        tokens = tokenizer.tokenize(data[0],
                                    maxlen=maxlen,
                                    add_spatial_tokens=True)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        mapping = tokenizer.rematch(data[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        length = len(tokens)
        labels = np.zeros((len(target), length, length))

        for start, end, label in data[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]

                label_index = target.index(label)
                labels[label_index, start, end] = 1

        output = {"input_ids": input_ids, "labels": labels}
        return output

    def __len__(self):
        return len(self.data)


def main():

    train_dataset = NERDataset(train_data)
    trainer.train(model,
                  train_dataset=train_dataset,
                  valid_dataset=None,
                  collate_fn=bert_sequence_label_gp_collate_fn)


if __name__ == '__main__':
    main()
