from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, EarlyStoppingCallback
from itertools import product
from datetime import datetime
import torch

class BERTTrainer:
    def __init__(self, model_name, idx2tag, tag2idx, evaluator):
        self.model_name = model_name
        self.idx2tag = idx2tag
        self.tag2idx = tag2idx
        self.evaluator = evaluator
        
        self.config = AutoConfig.from_pretrained(
            model_name,
            id2label=idx2tag,
            label2id=tag2idx,
            num_labels=len(idx2tag),
            output_hidden_states=False,
            output_attentions=False,
        )

    def finetune_model(self, processor, tokenized_dataset, epochs=3, batch_size=16, 
                      lr=5e-5, scheduler='linear', beta1=0.9, beta2=0.99, 
                      epsilon=1e-6, decay=0.01, early_stop_patience=2, freeze=False, output_dir='./results'):
        
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, 
            config=self.config
        )
        
        if torch.cuda.is_available():
            model.cuda()
            
        if freeze:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in model.bert.encoder.layer[:3]:
                for param in layer.parameters():
                    param.requires_grad = False

        training_args = TrainingArguments(
            report_to='none',
            output_dir=output_dir,
            learning_rate=lr,
            lr_scheduler_type=scheduler,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            optim="adamw_torch",
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            adam_beta1=beta1,
            adam_beta2=beta2,
            adam_epsilon=epsilon,
            fp16=True
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stop_patience,
            early_stopping_threshold=0.0001
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['dev'],
            processing_class=processor.tokenizer,
            data_collator=processor.data_collator,
            compute_metrics=self.evaluator.compute_metrics,
            callbacks=[early_stopping_callback]
        )

        trainer.train()

        return model, trainer

    def grid_search(self, processor, tokenized_dataset, param_grid=None):
        results = []
        best_f1 = 0
        best_model = None
        best_params = None

        if param_grid:
            param_combinations = [dict(zip(param_grid.keys(), v))
                                for v in product(*param_grid.values())]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, params in enumerate(param_combinations, 1):
            print(f"\n{i}/{len(param_combinations)}")
            print(f"Params: {params}")

            try:
                model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name, 
                    num_labels=len(self.idx2tag), 
                    id2label=self.idx2tag, 
                    label2id=self.tag2idx
                )

                training_args = TrainingArguments(
                    report_to='none',
                    output_dir=f'./results_{timestamp}_{i}',
                    learning_rate=params['learning_rate'],
                    lr_scheduler_type=params['lr_scheduler_type'],
                    per_device_train_batch_size=params['batch_size'],
                    per_device_eval_batch_size=params['batch_size'],
                    num_train_epochs=params['num_train_epochs'],
                    weight_decay=params['weight_decay'],
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    optim="adamw_torch",
                    load_best_model_at_end=True,
                    metric_for_best_model='f1',
                    adam_beta1=params['adam_beta1'],
                    adam_beta2=params['adam_beta2'],
                    adam_epsilon=params['adam_epsilon'],
                    fp16=True
                )

                early_stopping_callback = EarlyStoppingCallback(
                    early_stopping_patience=2,
                    early_stopping_threshold=0.0001
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset['train'],
                    eval_dataset=tokenized_dataset['dev'],
                    processing_class=processor.tokenizer,
                    data_collator=processor.data_collator,
                    compute_metrics=self.evaluator.compute_metrics,
                    callbacks=[early_stopping_callback]
                )

                train_result = trainer.train()
                eval_result = trainer.evaluate()

                trial_results = {
                    'parameters': params,
                    'eval_metrics': eval_result,
                    'train_metrics': {
                        'train_runtime': train_result.metrics['train_runtime'],
                        'train_samples_per_second': train_result.metrics['train_samples_per_second']
                    }
                }
                results.append(trial_results)

                if eval_result['eval_f1'] > best_f1:
                    best_f1 = eval_result['eval_f1']
                    best_model = model
                    best_params = params

            except Exception as e:
                print(f"Error {i}: {e}")
                continue

        print(f"Best F1: {best_f1}")
        print(f"Best parameters: {best_params}")

        return best_model, best_params, results