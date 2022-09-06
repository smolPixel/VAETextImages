from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from multiprocessing import cpu_count

class Bert_Classifier(pl.LightningModule):

    def __init__(self, argdict):
        super().__init__()
        self.argdict=argdict
        self.trained=False
        try:
            # self.tokenizer = BertTokenizer.from_pretrained('Models/bert_labellers_tokenizer.ptf')
            self.model = BertForSequenceClassification.from_pretrained(f'Models/{self.argdict["dataset"]}/bert_labeller', num_labels=len(self.argdict['categories']))
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # acc, confMatr = self.calculateAccuracyDev(test, self.model, self.tokenizer)
            # print(f"Model has already been trained with an accuracy of {acc}")
            # print(confMatr)
            self.trained=True
            print("Loaded Model")
        except:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.argdict['num_classes'])
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False

        # print(self.model)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        return optimizer


    def on_train_batch_start(self, batch, batch_idx):
        text_batch = batch['sentence']
        encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)  #
        batch['input_ids']=input_ids
        batch['attention_mask']=attention_mask

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'].long()#
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        # print(outputs)
        loss = outputs.loss
        results = torch.argmax(torch.log_softmax(outputs['logits'], dim=1), dim=1)

        self.log("Train_Acc", accuracy_score(results.cpu(), batch['label'].cpu()), on_epoch=True, prog_bar=True)
        self.log("Loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        text_batch = batch['sentence']
        encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)  #
        batch['input_ids']=input_ids
        batch['attention_mask']=attention_mask


    def validation_step(self, batch, batch_idx):
        # text_batch = batch['sentence']
        # encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        # input_ids = encoding['input_ids'].to(self.device)
        # attention_mask = encoding['attention_mask'].to(self.device)#
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        # print(encoding)
        labels = batch['label']#
        outputs = self.model(input_ids, attention_mask=attention_mask)
        results = torch.argmax(torch.log_softmax(outputs['logits'], dim=1), dim=1)

        self.log("Val_Acc", accuracy_score(results.cpu(), batch['label'].cpu()), on_epoch=True, prog_bar=True)
        # self.model.train()
        return outputs.loss

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        text_batch = batch['sentence']
        encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)  #
        batch['input_ids']=input_ids
        batch['attention_mask']=attention_mask


    def test_step(self, batch, batch_idx):
        input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        # print(encoding)
        labels = batch['label']#
        outputs = self.model(input_ids, attention_mask=attention_mask)
        results = torch.argmax(torch.log_softmax(outputs['logits'], dim=1), dim=1)
        # pred[start:start + 64] = results
        # Y[start:start + 64] = batch['label']
        # start = start + 64
        self.log("Test_Acc", accuracy_score(results.cpu(), batch['label'].cpu()), on_epoch=True, prog_bar=True)
        # self.model.train()
        return outputs.loss

    def init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(self.argdict['categories']))
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False

    def train_model(self, training_set, dev_set):
        # print("WARNING REIMPLEMENT EARLY STOPPING")
        # self.best_model=ModelCheckpoint(dirpath='Temp', monitor='Val_Acc', mode='max', filename=f'best_{self.argdict["dataset"]}_{self.argdict["algo"]}', save_top_k=1, every_n_epochs=1)

        self.trainer = pl.Trainer(gpus=self.argdict['gpus'], max_epochs=self.argdict['num_epochs_classifier'], precision=16)#, persistent_workers=True)#, enable_checkpointing=False)
        # trainer=pl.Trainer(max_epochs=self.argdict['num_epochs'])
        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.argdict['batch_size_classifier'],
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=self.argdict['batch_size_classifier'],
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        if not self.trained:
            self.trainer.fit(self, train_loader, dev_loader)
            self.model.save_pretrained(f'Models/{self.argdict["dataset"]}/bert_labeller')

        final = self.trainer.test(self, dev_loader)
        print(final)
        return final[0]['Test_Acc'], self.current_epoch


    def predict(self, dataset):
        # ds = Transformerdataset(dataset, split='train', labelled_only=False)
        # Test
        # print(len(dataset))
        # fds
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        Confidence = torch.zeros(len(dataset))
        start = 0
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                text_batch = batch['sentence']
                encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']
                # print(encoding)
                labels = batch['label']
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=torch.zeros_like(labels))
                results = torch.max(torch.softmax(outputs['logits'], dim=1), dim=1)
                # print("FuckYouPytorch")
                # print(results)
                # print(results[0])
                # print(results[0][0])
                Confidence[start:start + 64] = results[0]
                start = start + 64

        # print(Confidence)
        # # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)
        # dataset['confidence']=Confidence
        return Confidence

    def label(self, texts):
        #Predict the label of text
        with torch.no_grad():
            text_batch = texts
            encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            # print(encoding)
            # labels = batch['label']
            outputs = self.model(input_ids, attention_mask=attention_mask)
            results = torch.max(torch.softmax(outputs['logits'], dim=1), dim=1)
            Confidence= results[0]

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)
        return results[1], results[0]

    def get_logits(self, texts, encoded=None):
        with torch.no_grad():
            text_batch = texts
            encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            # print(encoding)
            # labels = batch['label']
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)
        return outputs['logits']

    def get_logits_from_tokenized(self, tokens, am):
        with torch.no_grad():
            # text_batch = texts
            # encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            # input_ids = encoding['input_ids']
            # attention_mask = encoding['attention_mask']
            # print(encoding)
            # labels = batch['label']
            # print(tokens.shape)
            outputs = self.model(inputs_embeds=tokens, attention_mask=am)

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)
        return outputs['logits']

    def get_grad(self, dataset):
        self.model.eval()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        Confidence = torch.zeros(len(dataset))
        start = 0
        loss=0
        for i, batch in enumerate(data_loader):
            text_batch = batch['sentence']
            encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            # print(encoding)
            labels = batch['label']
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss+=outputs['loss']
            # results = torch.max(torch.softmax(outputs['logits'], dim=1), dim=1)
            # print(results)
            # fds
            # print("FuckYouPytorch")
            # print(results)
            # print(results[0])
            # print(results[0][0])
            # Confidence[start:start + 64] = results[0]
            # start = start + 64
        return loss