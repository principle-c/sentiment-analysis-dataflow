import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import apache_beam as beam
import pandas as pd

def get_source_query(sample_size):
    from configparser import ConfigParser
    config_object = ConfigParser()
    config_object.read("config.ini")

    session_info = config_object["SESSION_INFO"]
    bigquery_config = config_object["BIGQUERY_CONFIG"]
    db_name = session_info["session_name"]
    bq_project = bigquery_config["project_name"]
    bq_account = bigquery_config["account_name"]
    bq_auth = bigquery_config["google_credential_path"]

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_auth

    dataset_name = bq_account + '_' + db_name
    base_table_id = bq_project + '.' + dataset_name + '.' + 'base_tweets'
    pred_table_id = bq_project + '.' + dataset_name + '.' + 'tweet_predictions'
    SOURCE_QUERY = f"""
                 SELECT base.tweet_id, clean_text
                 FROM {base_table_id} base
                 LEFT JOIN (SELECT tweet_id FROM {pred_table_id}) pt ON base.tweet_id = pt.tweet_id
                 WHERE pt.tweet_id IS NULL
                 ORDER BY created_at ASC
                 LIMIT {sample_size}
             """
    return SOURCE_QUERY

def get_sample_size_desc(sample_size):
    desc = '({}{} Rows)'
    if sample_size >= 1000000:
        desc = desc.format(sample_size/1000000.0,'M')
    elif sample_size >= 1000:
        desc = desc.format(sample_size /1000.0, 'K')
    else:
        desc = desc.format(sample_size, '')
    return desc

def download_blob(bucket_name=None, source_blob_name=None, project=None, destination_file_name=None):
    from google.cloud import bigquery, storage

    storage_client = storage.Client(project)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


class TweetDataset(Dataset):
    def __init__(self, reviews, targets, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.max_len = max_len
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          truncation=True,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


class Predict_bert(beam.DoFn):
    def __init__(self, project=None, bucket_name=None, model_path=None, destination_file_name=None):
        self._model = None
        self._project = project
        self._bucket_name = bucket_name
        self._model_path = model_path
        self._destination_file_name = destination_file_name

    def setup(self):
        download_blob(bucket_name=self._bucket_name,
                      source_blob_name=self._model_path,
                      project=self._project,
                      destination_file_name=self._destination_file_name)
        # unpickle ktrain model
        # # Load model json file
        # json_file = open(self._destination_config_name, 'r')
        #
        # # Load Ktrain preproc file
        # features = pickle.load(open(self._destination_preproc_name, 'rb'))
        #
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        #
        # loaded_model.load_weights(self._destination_h5_name)
        # print("Model Loaded from disk")
        #
        # # compile and evaluate loaded model
        # loaded_model.compile(optimizer='AdamW',
        #                      loss='categorical_crossentropy', metrics=['acc'])
        # #return loaded_model, features
        #model_path = self._destination_file_name.split('.')[0]


        model = SentimentClassifier(3)
        model.load_state_dict(torch.load(self._destination_file_name))
        model.eval()

        self._model = model

    def process(self, element):
        """Predicting using developed model"""
        # input_dat = {k: element[k] for k in element.keys() if k not in ['customerID']}
        # tmp = np.array(list(i for i in input_dat.values()))
        # tmp = tmp.reshape(1, -1)
        tweet = element["clean_text"]
        df = pd.DataFrame({'text': [tweet], 'target': 0})

        ds = TweetDataset(
            reviews=df.text.to_numpy(),
            targets=df.target.to_numpy(),
            max_len=48
        )
        inputs = {
            'input_ids': ds[0]['input_ids'].reshape(1, 48),
            'attention_mask': ds[0]['attention_mask'].reshape(1, 48)
        }
        out = self._model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        _, preds = torch.max(out, dim=1)
        pred = str(int(preds[0]))
        element["prediction"] = pred
        output = {k: element[k] for k in element.keys() if k in ['tweet_id', 'prediction']}
        output['tweet_id'] = str(output['tweet_id'])
        print([output])
        return [output]