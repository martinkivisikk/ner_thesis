import numpy as np
from nervaluate import Evaluator
import evaluate
import os
import json
import datetime

class BERTEvaluator:
    def __init__(self, all_tags, ner_tags=['EVE', 'GEP', 'LOC', 'MUU', 'ORG', 'PER', 'PROD', 'UNK']):
        self.all_tags = all_tags
        self.ner_tags = ner_tags
        self.seqeval = evaluate.load("seqeval")

    def compute_metrics(self, p):
        # Kasutatakse treenimisel, f-skoor sõnapõhiselt seqeval järgi, infoks märgendipõhised tulemused nervaluate abil
        
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.all_tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.all_tags[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)

        metrics = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        # märgendikaupa tulemused ka
        for tag, values in results.items():
            if isinstance(values, dict):
                metrics[f"{tag} Precision"] = values["precision"]
                metrics[f"{tag} Recall"] = values["recall"]
                metrics[f"{tag} F1"] = values["f1"]
                metrics[f"{tag} Number"] = values["number"]

        return metrics

    
    def get_predictions(self, dataset, trainer):
        # Kasutab treenimisel kasutatud 'trainer' objekti, et teha andmestikul ennustused
        # Tagastab ennustuste ja tegelike märgendite listid
        predictions, labels, _ = trainer.predict(dataset)
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.all_tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.all_tags[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return true_predictions, true_labels

    def get_seqeval_results(self, predictions, actual):
        # Evaluate/Seqeval abil sõnapõhine hindamine
        return self.seqeval.compute(predictions=predictions, references=actual)

    def get_nervaluate_results(self, predictions, actual):
        evaluator = Evaluator(actual, predictions, tags=self.ner_tags, loader="list")
        results, results_by_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
        return results, results_by_tag, result_indices, result_indices_by_tag

    def evaluate_model(self, test_dataset, trainer):
        print('Hindan testandmestikul..')
        predictions = self.get_predictions(test_dataset, trainer)
        seqeval_result = self.get_seqeval_results(*predictions)
        results, results_by_tag, result_indices, result_indices_by_tag = self.get_nervaluate_results(*predictions)
        return seqeval_result, results, results_by_tag

    def print_results(self, seqeval_result, nereval_result, nereval_by_tag):
        print("Seqeval tulemused")
        for key in seqeval_result:
            print(key, seqeval_result[key])

        print()
        print("Nervaluate tulemused")
        print("Strict", nereval_result['strict'])
        print("precision", nereval_result['strict']['precision'])
        print("recall", nereval_result['strict']['recall'])
        print("f1", nereval_result['strict']['f1'])

        for tag in nereval_by_tag:
            print(tag, nereval_by_tag[tag]['strict'])

    def evaluate_and_print(self, test_dataset, trainer):
        seqeval_result, nervaluate_result, nervaluate_by_tag = self.evaluate_model(test_dataset, trainer)
        self.print_results(seqeval_result, nervaluate_result, nervaluate_by_tag)
        return seqeval_result, nervaluate_result, nervaluate_by_tag
    
    def evaluation_to_json(self, nervaluate_strict_overall, nervaluate_by_tag, model_name, trained_on, evaluated_on, epochs=None):
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
        
        if trained_on:
            folder_name = f"results/{model_name}_{trained_on}"
        else:
            folder_name = f"results/{model_name}"

        timestamp = datetime.datetime.now().strftime("%Y-%d-%m_%H-%M")
        file_name = f"eval_on_{evaluated_on}_test_{timestamp}.json"

        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(formatted_output, f, indent=4)

        print(f"Salvestasin: {file_path}")
