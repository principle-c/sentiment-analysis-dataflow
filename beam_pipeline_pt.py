import apache_beam as beam
from apache_beam.options.pipeline_options import StandardOptions, GoogleCloudOptions, SetupOptions, PipelineOptions
from configparser import ConfigParser
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
from apache_beam.io.gcp.internal.clients import bigquery as beam_bq
import os
import re
import time
import logging
import google.cloud.logging  # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging
import traceback
import argparse
from pipeline_package import pipe_module as pm

def runflow(sample_size, runner, argv=None):
    from pipeline_package import pipe_module as pm

    parser = argparse.ArgumentParser()
    known_args, pipeline_args = parser.parse_known_args(argv)

    config_object = ConfigParser()
    config_object.read("config.ini")

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


    source_query = pm.get_source_query(sample_size)

    sample_size_desc = pm.get_sample_size_desc(sample_size)

    # options = PipelineOptions(
    #     flags=[],
    #     project=bq_project,
    #     job_name='sentiment-prediction-job-pt',
    #     temp_location='gs://sentiment-model-ja/temp',
    #     staging_location='gs://sentiment-model-ja/temp',
    #     region='us-central1',
    #     max_num_workers=2)
    with beam.Pipeline(runner, argv=pipeline_args) as pipeline:
        table_schema = beam_bq.TableSchema()

        # Fields that use standard types.
        id_schema = beam_bq.TableFieldSchema()
        id_schema.name = 'tweet_id'
        id_schema.type = 'integer'
        id_schema.mode = 'required'
        table_schema.fields.append(id_schema)

        predict_schema = beam_bq.TableFieldSchema()
        predict_schema.name = 'prediction'
        predict_schema.type = 'string'
        predict_schema.mode = 'required'
        table_schema.fields.append(predict_schema)

        (
                pipeline
                | 'Read from BigQuery {}'.format(sample_size_desc) >> beam.io.ReadFromBigQuery(query=source_query, use_standard_sql=True)
                | 'predict' >> beam.ParDo(pm.Predict_bert(project=bq_project, bucket_name=model_bucket,
                                                       model_path=model_path,
                                                       destination_file_name=model_dest))
                | "Write data to BQ" >> beam.io.WriteToBigQuery(table='tweet_predictions', dataset=dataset_name, schema=table_schema,
                                                                project=bq_project,
                                                                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
        )

        job = pipeline.run()
        if runner == 'DataflowRunner':
            job.wait_until_finish()

if __name__=='__main__':
    runflow(1000,'DataflowRunner')