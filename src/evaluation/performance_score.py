#### import necessary modules ######
import bert_score
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from rouge_score import rouge_scorer
import datasets
import numpy as np
from nltk.translate import meteor_score
from packaging import version
import evaluate
import json
import os

if evaluate.config.PY_VERSION < version.parse("3.8"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
if NLTK_VERSION >= version.Version("3.6.4"):
    from nltk import word_tokenize


rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
meteor = evaluate.load('meteor')
bleu = evaluate.load("bleu")

##############################################################################################
def evaluate_reports(data_path):
    # Load the JSON files

    data = [json.loads(q) for q in open(os.path.expanduser(data_path), "r")]
    ans_dict = {item['question_id']: item['answer'] for item in data}
    gt_dict = {item['question_id']: item['gt_answer'] for item in data}

    results = {}

    for _id in ans_dict:
        if _id in gt_dict:
            reference = gt_dict[_id]
            prediction = ans_dict[_id]

            # Bert score
            _, R, F1_bert_score = bert_score.score([reference], [prediction], lang='en',rescale_with_baseline=True)

            # Rouge score
            rouge_scores = rouge.score(reference, prediction)

            # Meteor
            meteor_score = meteor.compute(predictions=[prediction], references=[reference])

            # bleu
            bleu_score = bleu.compute(predictions=[prediction], references=[reference])

            results[_id] = {
                'bert_score': float(F1_bert_score[0]),
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'meteor': meteor_score['meteor'],
                'bleu_score': bleu_score['bleu']
            }
            
            # results.append(result)
    return results

def mean_total_scores(results):

    mean_bert_score = 0
    mean_rouge1 = 0
    mean_rouge2 = 0
    mean_rougeL = 0
    mean_meteor = 0
    mean_bleu_score = 0
    
    len_results = len(results)
    for i in results:
        mean_bert_score += results[i]['bert_score']
        mean_rouge1 += results[i]['rouge1']
        mean_rouge2 += results[i]['rouge2']
        mean_rougeL += results[i]['rougeL']
        mean_meteor += results[i]['meteor']
        mean_bleu_score += results[i]['bleu_score']
    mean_bert_score /= len_results
    mean_rouge1 /= len_results
    mean_rouge2 /= len_results
    mean_rougeL /= len_results
    mean_meteor /= len_results
    mean_bleu_score /= len_results

    return mean_bert_score, mean_rouge1, mean_rouge2, mean_rougeL, mean_meteor, mean_bleu_score
    # return mean_rouge1, mean_rouge2, mean_rougeL, mean_meteor, mean_bleu_score

if __name__ == "__main__":
    data_path = "path/to/report/answer/json/file"
    results = evaluate_reports(data_path)
    mean_bert_score, mean_rouge1, mean_rouge2, mean_rougeL, mean_meteor, mean_bleu_score = mean_total_scores(results)

    print("bert_score: ",mean_bert_score)
    print("Rouge1: ", mean_rouge1)
    print("Rouge2: ", mean_rouge2)
    print("RougeL: ", mean_rougeL)
    print("meteor: ", mean_meteor)
    print("bleu_score: ", mean_bleu_score)

