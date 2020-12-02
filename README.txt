Run dataflow using the follow command. Update parameters as needed.

"""
python beam_pipeline_pt.py \
  --job_name pytorch-sentiment-job \
  --project my-api-project-167306 \
  --region us-central1 \
  --runner DataflowRunner \
  --setup_file ./setup.py \
  --staging_location gs://sentiment-model-ja/staging \
  --temp_location gs://sentiment-model-ja/temp \
  --max_num_workers=5 \
  --extra_package dist/pipeline_package-0.0.1.tar.gz
"""