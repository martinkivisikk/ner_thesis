from modules.data_processing import DatasetProcessor
from modules.bert_data_processing import BERTDataProcessor
from modules.bert_evaluator import BERTEvaluator
from modules.bert_trainer import BERTTrainer
ALL_TAGS = DatasetProcessor.ALL_TAGS
TAG2IDX = DatasetProcessor.TAG2IDX
IDX2TAG = DatasetProcessor.IDX2TAG

# NB! Kui EstBERT viskab CUDA errorit, siis on arvatavasti andmetes öäü asemel àùò. Sel juhul word_id lähevad üle 50000, aga confis on sõnastiku max suurus 50k.
# siis töötle andmed uuesti (from_json=False ja salvesta kasutades save_dataset_to_json)

def train_model(model_name, dataset_name, epochs=3, batch_size=16):
    if dataset_name.lower() == 'combined':
      ewt_processor = DatasetProcessor('ewt', from_json=True)
      edt_processor = DatasetProcessor('edt', from_json=True)
      ewt_dataset = DatasetProcessor.tag_to_id(ewt_processor.dataset, TAG2IDX)
      edt_dataset = DatasetProcessor.tag_to_id(edt_processor.dataset, TAG2IDX)
      dataset = DatasetProcessor.combine_datasetdicts(ewt_dataset, edt_dataset)
    elif dataset_name.lower() in ['ewt', 'edt']:
      processor = DatasetProcessor(dataset_name, from_json=True)
      dataset = DatasetProcessor.tag_to_id(processor.dataset, TAG2IDX)

    print(f'{dataset_name.upper()} andmestik laetud')
    bert_processor = BERTDataProcessor(model_name)
    evaluator = BERTEvaluator(all_tags=ALL_TAGS)

    tokenized_dataset = bert_processor.tokenize_dataset(dataset)

    trainer = BERTTrainer(model_name=model_name, idx2tag=IDX2TAG, tag2idx=TAG2IDX, evaluator=evaluator)

    model, model_trainer = trainer.finetune_model(processor=bert_processor, tokenized_dataset=tokenized_dataset, epochs=epochs, batch_size=batch_size, early_stop_patience=3, output_dir=f'./results/{model_name.split("/")[1]}/{dataset_name}')

    results = evaluator.evaluate_and_print(tokenized_dataset['test'], model_trainer)

    return model, model_trainer, results

def main():
    model_names = ["tartuNLP/EstRoBERTa", "tartuNLP/EstBERT"]
    dataset_names = ['ewt', 'edt', 'combined']

    #train_model("tartuNLP/ESTRoBERTa", 'ewt', epochs=3, batch_size=32)

if __name__ == '__main__':
    main()
