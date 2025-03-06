from transformers import AutoTokenizer, DataCollatorForTokenClassification

class BERTDataProcessor:
    def __init__(self, model_name, max_length=128, padding="max_length", truncation="longest_first", do_lower_case=False, use_fast=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length, padding=padding, truncation=truncation, do_lower_case=do_lower_case, use_fast=use_fast)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def tokenize_and_align_labels(self, sent, label_all=True):
        # https://huggingface.co/docs/transformers/en/tasks/token_classification
        # Sisend: üks lause
        # Väljund: tokenizeri abil sõnestatud lause, sõnastik sisaldab input_ids, token_type_ids, attention_mask ja labels
        # input_ids on lause 'arvulisel' kujul, igale ID-le vastab mingi sõna (teisendamine funktsiooni tokenizer.convert_ids_to_tokens(input_ids) abil)
        # labels on märgendite list, kus märgendid kattuvad tokenizeri abil sõnestatud lausega
        tokenized_inputs = self.tokenizer(
            sent['tokens'],
            is_split_into_words=True,
            max_length=128,
            padding="max_length",
            truncation="longest_first"
        )

        labels = []
        word_ids = tokenized_inputs.word_ids()
        prev_word = None

        for word in word_ids:
          if word is None:
            label = -100
          elif word != prev_word:
            label = sent['tags'][word]
          else:
            label = sent['tags'][word] if label_all else -100
          labels.append(label)
          prev_word = word

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_dataset(self, dataset, label_all=True):
        return dataset.map(self.tokenize_and_align_labels, label_all)

    def get_data_collator(self):
        return self.data_collator