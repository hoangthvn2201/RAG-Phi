import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()
import debugpy

#https://github.com/richard-peng-xia/RULE/blob/main/llava/eval/model_vqa_iuxray.py
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



def main(model_path, image_folder, question_file, answer_file):
    # init llava modelmodel
    disable_torch_init()
    model_base = None 
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    
    # format question file
    num_chunks = 1
    chunk_idx = 0 
    conv_mode = "vicuna_v1"
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions,num_chunks, chunk_idx)

    ans_file = open(answer_file, "w")

    for line in tqdm(questions):
        idx = line["id"]
        image_file = line["image"]
        qs = ""
        gt_answer=line["report"]
        if "reference_reports" not in line: #determine when to use retriever
            # continue
            cur_prompt = "You are a professional radiologist. Please generate a report based on the image"
            qs=cur_prompt
        
        else:
            reference_report=line["reference_reports"]
            # print(reference_report)
            if not isinstance(reference_report, list):
                topk=1
                reference_report=[reference_report]
                formatted_reference_report=reference_report[0]
            else:
                topk=len(reference_report)
                formatted_reference_report=""
                for i in range(topk):
                    formatted_reference_report += f"{i + 1}. {reference_report[i]} "
                # print(formatted_reference_report)
            # cur_prompt = qs
            appendix_1=f"You are a professional radiologist. You are provided with an X-ray image and {topk} reference report(s): "
            appendix_2="\nPlease generate a report based on the image. It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, but should only be used for reference and comparison. Please only include the content of the report in your response."
            cur_prompt = appendix_1 + formatted_reference_report +"\n"+ appendix_2 +qs
            # print(cur_prompt)
            qs=cur_prompt
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None ,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "answer": outputs,
                                "gt_answer":gt_answer,
                                "image":image_file,
                                "image_id":image_file,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":

    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    image_folder = "path-to-iuxray-image-folder"
    question_file = "path-to-report-test-file"
    answer_file = "answer_file.jsonl"

    main(model_path, image_folder, question_file, answer_file)