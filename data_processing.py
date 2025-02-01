import os
import json
from estnltk.converters.conll import conll_importer
from datasets import Dataset, DatasetDict

ALL_TAGS = ['O',
            'B-Eve', 'I-Eve',
            'B-Gep', 'I-Gep',
            'B-Loc', 'I-Loc',
            'B-Muu', 'I-Muu',
            'B-Org', 'I-Org',
            'B-Per', 'I-Per',
            'B-Prod', 'I-Prod',
            'B-Unk', 'I-Unk']

# https://github.com/Kyubyong/nlp_made_easy/blob/master/Pos-tagging%20with%20Bert%20Fine-tuning.ipynb
# Sõnastikud, kus on vastavuses arv:märgend ja vastupidi, näiteks tag2idx sõnastikus 'B-Eve' -> 1 ning idx2tag sõnastikus siis 1 -> 'B-Eve'
TAG2IDX = {tag: idx for idx, tag in enumerate(ALL_TAGS)}
IDX2TAG = {idx: tag for idx, tag in enumerate(ALL_TAGS)}

def get_dataset_paths(dataset: str) -> dict:
    # Sisend: andmestiku nimi sõnena (ewt/edt)
    # Väljund: sõnastik, mis sisaldab train/dev/test failiteid
    dataset_dir = os.path.join('data', dataset)
    files = os.listdir(dataset_dir)

    paths = {}
    for split in ['train', 'dev', 'test']:
        matching_file = next(f for f in files if f'-ud-{split}.' in f)
        paths[split] = os.path.join(dataset_dir, matching_file)

    return paths

def preprocess(dataset_path: str) -> list:
    # Sisend: andmestiku failitee sõnena
    # Väljund: List, mis sisaldab parsitud lauseid
    # Iga lause on paaride list kujul [(w0, t0), (w1, t1), ..., (wn, tn)], kus w tähistab sõna ja t sõnale vastavat märgendit.

    dataset = conll_importer.conll_to_text(file=dataset_path)
    parsed_sents = []
    known_tags = ['B-Eve', 'I-Eve',
                  'B-Gep', 'I-Gep',
                  'B-Loc', 'I-Loc',
                  'B-Muu', 'I-Muu',
                  'B-Org', 'I-Org',
                  'B-Per', 'I-Per',
                  'B-Prod', 'I-Prod',
                  'B-Unk', 'I-Unk']

    # Kuna andmestikus on üksikud vead, aga on enam-vähem selge, mida tegelikult mõeldi, siis teeme vastavad parandused.
    corrections = {
      'B-OrgSpaceAfter': 'B-Org',
      'B_Gep': 'B-Gep',
      'i-Prod': 'I-Prod',
      'Org': 'B-Org',
      'Per': 'B-Per',
      'BäOrg': 'B-Org',
      'B.Prod': 'B-Prod',
      'I-per': 'I-Per'
    }

    for sent in dataset.sentences:
        parsed_sent = []
        for word, misc in zip(sent.words, sent.conll_syntax.misc):
            tag = 'O'
            if misc:
                if 'NE' in misc:
                  if misc['NE'] in known_tags:
                    tag = misc['NE']
                  else:
                    # Kaks üksikut juhtu, kus kahe elemendi pikkune nimeüksus oli märgendatud (_, Per), (_, Per) või (_, Org), (_, Org)
                    if parsed_sent[-1][1] == 'B-Org' and misc['NE'] == 'Org':
                      tag = 'I-Org'
                    if parsed_sent[-1][1] == 'B-Per' and misc['NE'] == 'Per':
                      tag = 'I-Per'
                    else:
                      tag = corrections[misc['NE']]
            pair = (word.text, tag)
            #print(f"({word.text}, {tag})")
            parsed_sent.append(pair)
        parsed_sents.append(parsed_sent)

    return parsed_sents

def split_to_token_and_tag(sents, tag2idx):
  # Sisend: parsitud laused ja tag2idx sõnastik
  # Väljund: Sõnastike list
  # Sõnastik sisaldab kolme elementi: lause ID täisarvuna, märgendite list arvulisel kujul ning sõnade list

  res = {}
  #res = []
  for i, sent in enumerate(sents):
    tags = [tag2idx[tag] for _, tag in sent]
    words = [word for word, _ in sent]
    res[i] = {
        'id': i,
        'tags': tags,
        'tokens': words
    }

  return res

def transform_set(data):
  transformed = {
      "id": [v["id"] for v in data.values()],
      "tags": [v["tags"] for v in data.values()],
      "tokens": [v["tokens"] for v in data.values()]
  }
  ds = Dataset.from_dict(transformed)
  return ds

def process_all(train_sents, dev_sents, test_sents, tag2idx):
  # Sisend: train/dev/test lausete listid ja tag2idx sõnastik
  # Väljund: töödeldud andmestik
  train = split_to_token_and_tag(train_sents, tag2idx)
  dev = split_to_token_and_tag(dev_sents, tag2idx)
  test = split_to_token_and_tag(test_sents, tag2idx)

  train_ds = transform_set(train)
  dev_ds = transform_set(dev)
  test_ds = transform_set(test)

  dataset = DatasetDict({
      'train': train_ds,
      'dev': dev_ds,
      'test': test_ds
  })

  return dataset

def save_split_to_json(split_data, output_path):
    serializable_data = []

    for item in split_data:
        data_dict = {
            'id': item['id'],
            'tags': item['tags'],
            'tokens': item['tokens']
        }
        serializable_data.append(data_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

def save_dataset_to_json(dataset, name=''):
  try:
    for split_name, split_data in dataset.items():
      output_path = f'data/{name}/{split_name}.json'
      save_split_to_json(split_data, output_path)
  except Exception as e:
    print(f"Error: {e}")

def load_split_from_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_dict = {
        'id': [],
        'tags': [],
        'tokens': []
    }

    for item in data:
        dataset_dict['id'].append(item['id'])
        dataset_dict['tags'].append(item['tags'])
        dataset_dict['tokens'].append(item['tokens'])

    return Dataset.from_dict(dataset_dict)

def load_dataset_from_json(name=''):
  try:
    dataset = DatasetDict()
    for split_name in ['train', 'dev', 'test']:
      input_path = f'data/{name}/{split_name}.json'
      dataset[split_name] = load_split_from_json(input_path)
    return dataset
  except Exception as e:
    print(f"Error: {e}")

def load_and_process_dataset(dataset_name: str) -> DatasetDict:
   paths = get_dataset_paths(dataset_name)

   dev_sents = preprocess(paths['dev'])
   train_sents = preprocess(paths['train'])
   test_sents = preprocess(paths['test'])

   return process_all(train_sents, dev_sents, test_sents, TAG2IDX)

def combine_datasetdicts(dataset1, dataset2):
  combined = DatasetDict()

  for split in ['train', 'dev', 'test']:
    combined_dict = {
        'id': [],
        'tags': [],
        'tokens': []
    }

    for i, item in enumerate(dataset1[split]):
      combined_dict['id'].append(i)
      combined_dict['tags'].append(item['tags'])
      combined_dict['tokens'].append(item['tokens'])
    offset = len(dataset1[split])
    for i, item in enumerate(dataset2[split]):
      combined_dict['id'].append(i + offset)
      combined_dict['tags'].append(item['tags'])
      combined_dict['tokens'].append(item['tokens'])
    combined[split] = Dataset.from_dict(combined_dict)
  return combined

def load_all(from_json=True):
  try:
    if from_json:
      ewt_dataset = load_dataset_from_json('ewt')
      edt_dataset = load_dataset_from_json('edt')
      combined_dataset = load_dataset_from_json()
    else:
      ewt_dataset = load_and_process_dataset('ewt')
      edt_dataset = load_and_process_dataset('edt')
      combined_dataset = combine_datasetdicts(ewt_dataset, edt_dataset)
    
    return ewt_dataset, edt_dataset, combined_dataset
  except:
    print(f"error andmete laadimisel from_json={from_json}")