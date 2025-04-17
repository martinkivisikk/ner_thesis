import csv
import os
import datetime
import json
from transformers import pipeline, AutoConfig

ALL_TAGS = ['O',
            'B-EVE', 'I-EVE',
            'B-GEP', 'I-GEP',
            'B-LOC', 'I-LOC',
            'B-MUU', 'I-MUU',
            'B-ORG', 'I-ORG',
            'B-PER', 'I-PER',
            'B-PROD', 'I-PROD',
            'B-UNK', 'I-UNK']

def write_to_tsv(sents, true_tags, pred_tags, model_name, evaluated_on):
  folder_name = f"results/{model_name}"
  os.makedirs(folder_name, exist_ok=True)
  timestamp = datetime.datetime.now().strftime("%Y-%d-%m_%H-%M")
  file_name = f"predictions_{evaluated_on}_test_{timestamp}.tsv"
  file_path = os.path.join(folder_name, file_name)

  with open(file_path, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['word', 'true_tag', 'pred_tag'])
    for sent, true, pred in zip(sents, true_tags, pred_tags):
      for word, true_tag, pred_tag in zip(sent, true, pred):
        writer.writerow([word, true_tag, pred_tag])
      writer.writerow([])

def evaluation_to_json(nervaluate_strict_overall, nervaluate_by_tag, model_name, trained_on, evaluated_on, epochs=None):
        results = {}
        results["strict"] = nervaluate_strict_overall
        for key in nervaluate_by_tag:
            results[key] = nervaluate_by_tag[key]['strict']

        formatted_output = {
            "model": model_name,
            "trained_on": trained_on,
            "epochs": epochs,
            "evaluated_on": evaluated_on,
            "results": results
        }

        folder_name = f"results/{model_name}"
        
        timestamp = datetime.datetime.now().strftime("%Y-%d-%m_%H-%M")
        file_name = f"eval_on_{evaluated_on}_test_{timestamp}.json"

        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(formatted_output, f, indent=4)

        print(f"Salvestasin: {file_path}")

def get_supported_entities(model_name):
  config = AutoConfig.from_pretrained(model_name)
  id2label = config.id2label

  supported_entities = set()
  for label in id2label.values():
    if label != 'O':
      entity_type = label[2:] if label.startswith(('B-', 'I-')) else label
      supported_entities.add(entity_type)

  return supported_entities

def predict_tags(model_name, data, is_roberta_model=False):
  ner = pipeline('ner', model=model_name, aggregation_strategy=None)
  predictions = []

  for item in data:
    sentence = " ".join(item['tokens'])
    prediction = ner(sentence)

    spans = []
    current = 0
    for token in item['tokens']:
      spans.append((current, current + len(token)))
      current += len(token) + 1

    prediction_tags = ['O'] * len(item['tokens'])

    entities = []
    current_entity = None

    for pred in prediction:
      tag = pred['entity']
      word = pred['word']
      start = pred['start']
      end = pred['end']
      
      if is_roberta_model:
        if (not word.startswith('▁')) or word in ['<s>', '</s>', '<pad>']:
          continue
      else:
        if word.startswith('##') or word in ['[CLS]', '[SEP]', '[PAD]']:
          continue

      if tag.startswith('B-') or (tag == 'O' and current_entity is not None):
        if current_entity is not None:
          entities.append(current_entity)
          current_entity = None

      if tag.startswith('B-'):
        entity_type = tag[2:]
        current_entity = {'type': entity_type, 'start': start, 'end': end}

      elif tag.startswith('I-'):
        entity_type = tag[2:]
        if current_entity is not None and current_entity['type'] == entity_type:
          current_entity['end'] = end

    if current_entity is not None:
      entities.append(current_entity)

    for entity in entities:
      entity_start = entity['start']
      entity_end = entity['end']
      entity_type = entity['type']

      first = True
      for i, (start, end) in enumerate(spans):
        if start < entity_end and end > entity_start: #if start <= entity_end and end >= entity_start:
          if first:
            prediction_tags[i] = f'B-{entity_type}'
            first = False
          else:
            prediction_tags[i] = f'I-{entity_type}'

    predictions.append(prediction_tags)

  return predictions