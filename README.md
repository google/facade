# FACADE: Fast and Accurate Contextual Anomaly DEtection

FACADE is an enterprise-security anomaly detection system developed by Google.
It is a high-precision deep-learning-based machine learning system used in
a number of applications across Google. It is used as a last line of defense
against insider threats, as an ACL recommendation system, and as a way to
detect account compromise.

This repository serves as a reference implementation for the concepts presented
at BlackHat 2025. The primary purpose of this code is to provide a concrete
example of the ideas discussed in the talk. Please be aware that FACADE is
released on a best-effort basis. There are no guarantees of ongoing support,
bug fixes, or future feature development. You are welcome to fork the repository
and adapt the code for your own needs.

For more details see our paper at: https://arxiv.org/abs/2412.06700

BlackHat 2025 slides: https://elie.net/facade

## Setup

FACADE requires Python >=3.10. The easiest way to set up Facade's dependencies
is with pip and virtual env:

```shell
python3 -m venv env
source env/bin/activate
pip install -r requirements.in
```

## Sample data

FACADE contains synthetic sample data in 
[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format in `samples.zip`.
This data replicates employee activity of a small business, with logs for document
accesses (actions) and code reviews (contexts). Below are the commands to train
and run a model on this sample data:

```shell
# Unzip the sample data
unzip sample.zip

# Generate training data
python -m batch.dataset_maker_main \
  --directive=sample/directive.textproto \
  --start_time="2024-04-01 00:00:00" \
  --end_time="2024-07-01 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --train_output=sample/train.tfrecord

# Generate validation data
python -m batch.dataset_maker_main \
  --directive=sample/directive.textproto \
  --start_time="2024-07-01 00:00:00" \
  --end_time="2024-07-08 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --validation_output=sample/validation.tfrecord

# Train the model
python -m model.train_main \
  --train_file=sample/train.tfrecord \
  --vocabulary_file=sample/vocabs.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model

# Evaluate the model
python -m model.train_main \
  --eval_file=sample/train.tfrecord \
  --vocabulary_file=sample/vocabs.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model \
  --is_evaluation_task

# Run inference
python -m batch.inference_main \
  --directive=sample/directive.textproto \
  --start_time="2024-07-08 00:00:00" \
  --end_time="2024-07-15 00:00:00" \
  --action_path=sample/action.tfrecord \
  --context_path=sample/context.tfrecord \
  --output_file=sample/scores.tfrecord \
  --model_config=sample/config.textproto \
  --model_dir=sample/model

# Read top scores
python -m batch.read_scores_main \
  --score_file=sample/scores.tfrecord
```

## Contributors

FACADE was developed by:
 - Alex Kantchelian
 - Casper Neo
 - Ryan Stevens
 - Hyungwon Kim
 - Zhaohao Fu
 - Sadegh Momeni
 - Birkett Huber
 - Elie Bursztein
 - Yanis Pavlidis
 - Senaka Buthpitiya
 - Martin Cochran
 - Massimiliano Poletto
 - Louis Li

Special thanks for their contributions to this open-source release:
 - Cem Topcuoglu
 - Aniket Anand
