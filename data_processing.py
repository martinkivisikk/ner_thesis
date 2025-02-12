import os
import json
import re
from estnltk.converters.conll import conll_importer
from estnltk import Text
from datasets import Dataset, DatasetDict

#NB! Kui EstBERT viskab CUDA errorit, siis on arvatavasti andmetes öäü asemel àùò. Sel juhul word_id lähevad üle 50000, aga confis on sõnastiku max suurus 50k.
# siis töötle andmed uuesti (from_json=False ja salvesta kasutades save_dataset_to_json)

class DatasetProcessor:
    ALL_TAGS = ['O', 
            'B-EVE', 'I-EVE', 
            'B-GEP', 'I-GEP', 
            'B-LOC', 'I-LOC', 
            'B-MUU', 'I-MUU', 
            'B-ORG', 'I-ORG', 
            'B-PER', 'I-PER', 
            'B-PROD', 'I-PROD', 
            'B-UNK', 'I-UNK']
    TAG2IDX = {tag: idx for idx, tag in enumerate(ALL_TAGS)}
    IDX2TAG = {idx: tag for idx, tag in enumerate(ALL_TAGS)}

    KNOWN_TAGS = ['B-Eve', 'I-Eve',
                    'B-Gep', 'I-Gep',
                    'B-Loc', 'I-Loc',
                    'B-Muu', 'I-Muu',
                    'B-Org', 'I-Org',
                    'B-Per', 'I-Per',
                    'B-Prod', 'I-Prod',
                    'B-Unk', 'I-Unk']
    # Kuna andmestikus on üksikud vead, aga on enam-vähem selge, mida tegelikult mõeldi, siis teeme vastavad parandused.
    CORRECTIONS = {
            'B-OrgSpaceAfter': 'B-Org',
            'B_Gep': 'B-Gep',
            'i-Prod': 'I-Prod',
            'Org': 'B-Org',
            'Per': 'B-Per',
            'BäOrg': 'B-Org',
            'B.Prod': 'B-Prod',
            'I-per': 'I-Per'}

    CHAR_MAP = {"à": "ä","ù": "ü","ò": "o"}
    PATTERN = re.compile("|".join(re.escape(k) for k in CHAR_MAP.keys()))
    
    def __init__(self, dataset_name: str, from_json: bool = True):
        self.dataset_name = dataset_name
        self.dataset_paths = self.get_dataset_paths()
        self.from_json = from_json
    
    def get_dataset_paths(self):
        # Sisend: andmestiku nimi sõnena (ewt/edt)
        # Väljund: sõnastik, mis sisaldab train/dev/test failiteid
        
        dataset_dir = os.path.join('data', self.dataset_name)
        files = os.listdir(dataset_dir)
        paths = {}
        for split in ['train', 'dev', 'test']:
            matching_file = next(f for f in files if f'-ud-{split}.' in f)
            paths[split] = os.path.join(dataset_dir, matching_file)
        return paths
    
    @classmethod
    def replace_chars(cls, text):
        return cls.PATTERN.sub(lambda m: cls.CHAR_MAP[m.group(0)], text)
    
    # def get_split_as_text(self, split_path:str):
    #     text = conll_importer.conll_to_text(file=split_path)
    #     return text
    
    # def get_all_splits_as_text(self):
    #     train_text = self.get_split_as_text(self.dataset_paths['train'])
    #     test_text = self.get_split_as_text(self.dataset_paths['test'])
    #     dev_text = self.get_split_as_text(self.dataset_paths['dev'])
    #     return train_text, test_text, dev_text

    def get_train_data_for_tagger(self):
        train_texts = self.get_split_as_texts_list(self.dataset_paths['train'])
        train_tags = self.get_all_tag_lists(train_texts)
        return train_texts, train_tags

    def get_test_data_for_tagger(self):
        test_texts = self.get_split_as_texts_list(self.dataset_paths['test'])
        test_tags = self.get_all_tag_lists(test_texts)
        return test_texts, test_tags

    def get_split_as_texts_list(self, split_path:str):
        texts = conll_importer.conll_to_texts_list(file=split_path)
        for text in texts:
          text.tag_layer(['morph_analysis'])
        return texts

    def get_tag_lists(self, text: Text):
        #Ühe Text objekti lausete märgendid listide listina
        parsed_sents = []
        for sent in text.sentences:
            tags = []
            
            for misc in sent.conll_syntax.misc:
                tag = 'O'
                if misc and 'NE' in misc:
                    ne_tag = misc['NE']
                    if ne_tag in self.KNOWN_TAGS:
                        tag = ne_tag
                    elif tags and tags[-1] == 'B-ORG' and ne_tag == 'Org':
                        tag = 'I-Org'
                    elif tags and tags[-1] == 'B-PER' and ne_tag == 'Per':
                        tag = 'I-Per'
                    else:
                        tag = self.CORRECTIONS.get(ne_tag, 'O')
                tags.append(tag.upper())

            parsed_sents.append(tags)
        return parsed_sents

    def get_all_tag_lists(self, texts):
        tag_lists = []
        for text in texts:
            tag_lists.append(self.get_tag_lists(text))
        return tag_lists

    def preprocess(self, dataset_path: str, use_lemmas=False):
        # Sisend: andmestiku failitee sõnena
        # Väljund: List, mis sisaldab parsitud lauseid
        # Iga lause on paaride list kujul [(w0, t0), (w1, t1), ..., (wn, tn)], kus w tähistab sõna ja t sõnale vastavat märgendit.
        dataset = conll_importer.conll_to_text(file=dataset_path)
        parsed_sents = []
        
        for sent in dataset.sentences:
            parsed_sent = []
            for word, misc in zip(sent.words, sent.conll_syntax.misc):
                tag = self._get_tag(misc, parsed_sent)
                if use_lemmas:
                    w = word.conll_syntax.lemma
                else:
                    w = word.text
                pair = (self.replace_chars(w), tag.upper())
                parsed_sent.append(pair)
            parsed_sents.append(parsed_sent)
                
        return parsed_sents
    
    def _get_tag(self, misc, parsed_sent):
        if not misc or 'NE' not in misc:
            return 'O'
        ne_tag = misc['NE']
        
        if ne_tag in self.KNOWN_TAGS:
            return ne_tag
        
        if parsed_sent:
            last_tag = parsed_sent[-1][1]
            if last_tag == 'B-ORG' and ne_tag == 'Org':
                return 'I-Org'
            if last_tag == 'B-PER' and ne_tag == 'Per':
                return 'I-Per'
            
        return self.CORRECTIONS.get(ne_tag, 'O')
    
    def split_to_token_and_tag(self, sents):
        # Sisend: parsitud laused
        # Väljund: Sõnastike list
        # Sõnastik sisaldab kolme elementi: lause ID täisarvuna, märgendite list arvulisel kujul ning sõnade list
        res = {}
        for i, sent in enumerate(sents):
            tags = [self.TAG2IDX[tag] for _, tag in sent]
            words = [word for word, _ in sent]
            res[i] = {
                'id': i,
                'tags': tags,
                'tokens': words
                }
        return res
    
    def transform_set(self, data):
        transformed = {
            "id": [v["id"] for v in data.values()],
            "tags": [v["tags"] for v in data.values()],
            "tokens": [v["tokens"] for v in data.values()]
            }
        return Dataset.from_dict(transformed)
    
    def process_all(self, use_lemmas=False):
        train_sents = self.preprocess(self.dataset_paths['train'], use_lemmas)
        dev_sents = self.preprocess(self.dataset_paths['dev'], use_lemmas)
        test_sents = self.preprocess(self.dataset_paths['test'], use_lemmas)
        
        train_ds = self.transform_set(self.split_to_token_and_tag(train_sents))
        dev_ds = self.transform_set(self.split_to_token_and_tag(dev_sents))
        test_ds = self.transform_set(self.split_to_token_and_tag(test_sents))
        
        return DatasetDict({'train': train_ds, 'dev': dev_ds, 'test': test_ds})
    
    def load_or_process(self, use_lemmas=False):
        return self.load_dataset_from_json() if self.from_json else self.process_all(use_lemmas)
    
    def save_dataset_to_json(self, dataset):
        os.makedirs(f'data/{self.dataset_name}', exist_ok=True)
        for split_name, split_data in dataset.items():
            with open(f'data/{self.dataset_name}/{split_name}.json', 'w', encoding='utf-8') as f:
                json.dump([{'id': item['id'], 'tags': item['tags'], 'tokens': item['tokens']} for item in split_data], f, ensure_ascii=False, indent=2)
    
    def load_dataset_from_json(self):
        dataset = DatasetDict()
        for split_name in ['train', 'dev', 'test']:
            with open(f'data/{self.dataset_name}/{split_name}.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset[split_name] = Dataset.from_dict({'id': [item['id'] for item in data], 'tags': [item['tags'] for item in data], 'tokens': [item['tokens'] for item in data]})
        return dataset
    
    @staticmethod
    def combine_datasetdicts(dataset1, dataset2):
        combined = DatasetDict()
        for split in ['train', 'dev', 'test']:
            combined_dict = {'id': [], 'tags': [], 'tokens': []}
            combined_dict['id'] = list(range(len(dataset1[split]) + len(dataset2[split])))
            combined_dict['tags'] = dataset1[split]['tags'] + dataset2[split]['tags']
            combined_dict['tokens'] = dataset1[split]['tokens'] + dataset2[split]['tokens']
            combined[split] = Dataset.from_dict(combined_dict)
        return combined
