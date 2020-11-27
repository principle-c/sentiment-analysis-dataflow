import apache_beam as beam
from apache_beam.options.pipeline_options import StandardOptions, GoogleCloudOptions, SetupOptions, PipelineOptions
from configparser import ConfigParser
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
from apache_beam.io.gcp.internal.clients import bigquery as beam_bq
import os
import re
import pandas as pd
import pandas_gbq
import time
import logging
import google.cloud.logging  # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging
import traceback
import pickle


def get_source_query(sample_size):

    config_object = ConfigParser()
    config_object.read("sentimentcode/config.ini")

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
        """Download pytorch model from GCS"""
        logging.info(
            "pytorch model initialisation {}".format(self._model_path))
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


def run(sample_size, runner, argv=None):
    config_object = ConfigParser()
    config_object.read("sentimentcode/config.ini")

    session_info = config_object["SESSION_INFO"]
    bigquery_config = config_object["BIGQUERY_CONFIG"]
    db_name = session_info["session_name"]
    bq_project = bigquery_config["project_name"]
    bq_account = bigquery_config["account_name"]
    bq_auth = bigquery_config["google_credential_path"]
    model_bucket = bigquery_config["model_bucket"]
    model_path = bigquery_config["model_path"]
    model_dest = bigquery_config["model_dest"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq_auth

    session_id = bq_account + '_' + db_name
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client, name=session_id)
    logging.getLogger().setLevel(logging.INFO)  # defaults to WARN
    setup_logging(handler)
    dataset_name = bq_account + '_' + db_name
    base_table_id = bq_project + '.' + dataset_name + '.' + 'base_tweets'
    pred_table_id = bq_project + '.' + dataset_name + '.' + 'tweet_predictions'
    # Construct a BigQuery client object.

    client = bigquery.Client()
    # create the tweet_predictions table if does not exist
    try:
        client.get_table(pred_table_id)  # Make an API request.
        msg = "Table {} already exists.".format(pred_table_id)
        print(msg)
        logging.info(msg)
    except NotFound:
        msg = "Table {} is not found. Creating table...".format(pred_table_id)
        print(msg)
        logging.info(msg)

        schema = [
            bigquery.SchemaField("tweet_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("prediction", "STRING", mode="REQUIRED"),
        ]

        table = bigquery.Table(pred_table_id, schema=schema)
        table = client.create_table(table)  # Make an API request.
        msg = f'Created table {pred_table_id}'
        logging.info(msg)


    source_query = get_source_query(sample_size)

    sample_size_desc = get_sample_size_desc(sample_size)

    options = PipelineOptions(
        flags=[],
        runner='DataflowRunner',
        project=bq_project,
        job_name='pytorch-sentiment-prediction-job',
        temp_location='gs://sentiment-ja/temp',
        staging_location='gs://sentiment-ja/temp',
        region='us-central1')
    with beam.Pipeline(runner, options=options) as pipeline:
        table_schema = beam_bq.TableSchema()

        # Fields that use standard types.
        id_schema = beam_bq.TableFieldSchema()
        id_schema.name = 'tweet_id'
        id_schema.type = 'integer'
        id_schema.mode = 'nullable'
        table_schema.fields.append(id_schema)

        predict_schema = beam_bq.TableFieldSchema()
        predict_schema.name = 'prediction'
        predict_schema.type = 'string'
        predict_schema.mode = 'required'
        table_schema.fields.append(predict_schema)

        (
                pipeline
                | 'Read from BigQuery {}'.format(sample_size_desc) >> beam.io.ReadFromBigQuery(query=source_query, use_standard_sql=True)
                | 'predict' >> beam.ParDo(Predict_bert(project=bq_project, bucket_name=model_bucket,
                                                       model_path='model/model.pt',
                                                       destination_file_name='model.pt'))
                | "Write data to BQ" >> beam.io.WriteToBigQuery(table='tweet_prediction', dataset=dataset_name, schema=table_schema,
                                                                project=bq_project,
                                                                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
        )

        job = pipeline.run()
        if runner == 'DataflowRunner':
            job.wait_until_finish()

if __name__ == '__main__':
    run(10,'DataflowRunner')