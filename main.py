import logging

from sentimentcode import beam_pipeline_pt

if __name__ == '__main__':
  # logging.getLogger().setLevel(logging.INFO)
  beam_pipeline_pt.run(10, 'DataflowRunner')

  """
python main.py \
  --job_name pytorch-sentiment-job-$USER \
  --project sentiment-ja \
  --region us-central1 \
  --runner DataflowRunner \
  --setup_file ./setup.py \
  --staging_location gs://sentiment-ja/staging \
  --temp_location gs://sentiment-ja/temp
"""