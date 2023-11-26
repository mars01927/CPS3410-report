import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, AdamW, BertForSequenceClassification, \
    get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import accuracy_score, matthews_corrcoef

from tqdm import tqdm, trange, tnrange, tqdm_notebook
import random
import os
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

SEED = 19

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv("../Materias/Twitter_Data.csv")
df = df.dropna()
df['category'].unique()
df['category'].value_counts()
df = df[~df['category'].isnull()]
df = df[~df['clean_text'].isnull()]
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df['category_1'] = labelencoder.fit_transform(df['category'])
df[['category', 'category_1']].drop_duplicates(keep='first')
df.rename(columns={'category_1': 'label'}, inplace=True)
sentences = df.clean_text.values

print("Distribution of data based on labels: ", df.label.value_counts())

MAX_LEN = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = [
    tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True, truncation=True) for
    sent in sentences]

labels = df.label.values

print("Actual sentence before tokenization: ", sentences[2])
print("Encoded Input from dataset: ", input_ids[2])


attention_masks = []
attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]
print(attention_masks[2])

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=41,
                                                                                    test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=41, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

train_data[0]

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)

lr = 2e-5
adam_epsilon = 1e-8

epochs = 3

num_warmup_steps = 0
num_training_steps = len(train_dataloader) * epochs

optimizer = AdamW(model.parameters(), lr=lr, eps=adam_epsilon,
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)  # PyTorch scheduler

train_loss_set = []
learning_rate = []

model.zero_grad()

for _ in tnrange(1, epochs + 1, desc='Epoch'):
    print("<" + "=" * 22 + F" Epoch {_} " + "=" * 22 + ">")
    batch_loss = 0

    for step, batch in enumerate(train_dataloader):
        model.train()

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        batch_loss += loss.item()

    avg_train_loss = batch_loss / len(train_dataloader)

    for param_group in optimizer.param_groups:
        print("\n\tCurrent Learning rate: ", param_group['lr'])
        learning_rate.append(param_group['lr'])

    train_loss_set.append(avg_train_loss)
    print(F'\n\tAverage Training loss: {avg_train_loss}')

    model.eval()

    eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0, 0, 0

    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():

            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits[0].to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        df_metrics = pd.DataFrame({'Epoch': epochs, 'Actual_class': labels_flat, 'Predicted_class': pred_flat})

        tmp_eval_accuracy = accuracy_score(labels_flat, pred_flat)
        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

        eval_accuracy += tmp_eval_accuracy
        eval_mcc_accuracy += tmp_eval_mcc_accuracy
        nb_eval_steps += 1

    print(F'\n\tValidation Accuracy: {eval_accuracy / nb_eval_steps}')
    print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy / nb_eval_steps}')

    from sklearn.metrics import confusion_matrix, classification_report


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        label2int = {
            "Negative": -1,
            "Neutral": 0,
            "Positive": 1
        }
        print(classification_report(df_metrics['Actual_class'].values, df_metrics['Predicted_class'].values,
                                    target_names=label2int.keys(), digits=len(label2int)))

