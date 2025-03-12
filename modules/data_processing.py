import os
import json
import re
from estnltk.converters.conll import conll_importer
from estnltk import Text
from datasets import Dataset, DatasetDict

#NB! Kui EstBERT viskab CUDA errorit, siis on arvatavasti andmetes öäü asemel àùò. Sel juhul word_id lähevad üle 50000, aga confis on sõnastiku max suurus 50k.
# siis töötle andmed uuesti (from_json=False ja salvesta kasutades save_dataset_to_json)

class DatasetProcessor:
    # peab jälgima, et all_tags, idx2tag ja mudeli.config.id2label/label2id ja andmestik on kooskõlas
    # hardcoded all_tags on paras jama tegelikult, pigem kasutad
    ALL_TAGS = ['O',
            'B-EVE', 'I-EVE',
            'B-GEP', 'I-GEP',
            'B-LOC', 'I-LOC',
            'B-MUU', 'I-MUU',
            'B-ORG', 'I-ORG',
            'B-PER', 'I-PER',
            'B-PROD', 'I-PROD',
            'B-UNK', 'I-UNK']

    ALL_TAGS.sort(reverse=True)
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
    # Kuna andmestikus on üksikud vead, aga on enam-vähem selge, mida tegelikult mõeldi, siis vastavad parandused.
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
        self.dataset = self.load_or_process()
        if self.dataset:
            self.train = self.dataset['train']
            self.dev = self.dataset['dev']
            self.test = self.dataset['test']

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
        # Asendab àùò äüo
        return cls.PATTERN.sub(lambda m: cls.CHAR_MAP[m.group(0)], text)

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
        # Sisendiks siiani parsitud lause, ja failist misc info
        # Tagastab sõnale vastava märgendi
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
        # Väljund: sõnastik, mis sisaldab ID'de listi, kahte listi lausete sõnade ja märgendite jaoks, iga listi element on omakorda list, mis sisaldab sõnu/märgendeid
        return {
            "id": list(range(len(sents))),
            #"tags": [[self.TAG2IDX[tag] for _, tag in sent] for sent in sents],
            "tags": [[tag for _, tag in sent] for sent in sents],
            "tokens": [[word for word, _ in sent] for sent in sents]
        }

    def process_all(self, use_lemmas=False):
        # Töötleb train/dev/test laused
        # Tagastab DatasetDict objekti, mis sisaldab train/dev/test Dataset objekte
        train_sents = self.preprocess(self.dataset_paths['train'], use_lemmas)
        dev_sents = self.preprocess(self.dataset_paths['dev'], use_lemmas)
        test_sents = self.preprocess(self.dataset_paths['test'], use_lemmas)

        return DatasetDict({
            'train': Dataset.from_dict(self.split_to_token_and_tag(train_sents)),
            'dev': Dataset.from_dict(self.split_to_token_and_tag(dev_sents)),
            'test': Dataset.from_dict(self.split_to_token_and_tag(test_sents))
        })

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

    @staticmethod
    def tag_to_id(dataset, tag2idx):
        def _map_tags(example):
            tag_ids = [tag2idx[tag] for tag in example['tags']]
            return {'tags': tag_ids}

        mapped_dataset = dataset.copy()
        for split in mapped_dataset:
            mapped_dataset[split] = mapped_dataset[split].map(_map_tags)

        return DatasetDict(mapped_dataset)

    def get_split_stats(self, split):
        stats = {
            'sentences': 0,
            'tokens': 0
        }
        for sentence in split:
            stats['sentences'] += 1
            for tag, word in zip(sentence['tags'], sentence['tokens']):
                stats['tokens'] += 1
                #actual_tag = self.IDX2TAG[tag]
                actual_tag = tag
                tag_prefix = actual_tag[:2] # B-, I-, O
                tag_suffix = actual_tag[2:]
                if tag_prefix == 'B-':
                    stats[tag_suffix] = stats.get(tag_suffix, 0) + 1

        return stats

    def get_all_stats(self):
        train_stats = self.get_split_stats(self.train)
        test_stats = self.get_split_stats(self.test)
        dev_stats = self.get_split_stats(self.dev)

        total_stats = {}

        for stats in [train_stats, test_stats, dev_stats]:
            for key, value in stats.items():
                total_stats[key] = total_stats.get(key, 0) + value

        return {
            "train": train_stats,
            "test": test_stats,
            "dev": dev_stats,
            "total": total_stats
        }
