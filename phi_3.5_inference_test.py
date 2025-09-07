import argparse
import os
import json
import math
import shortuuid
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(model_id, image_folder, question_file, answer_file, reference=False):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="eager"  # change if you have flash_attn
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=4
    )

    # Load dataset
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, 1, 0)

    ans_file = open(answer_file, "w")

    for line in tqdm(questions):
        idx = line["id"]
        image_file = line["image"]
        gt_answer = line["report"]

        images = []
        placeholder = ""
        images.append(Image.open(os.path.join(image_folder, image_file)))
        placeholder += f"<|image_{1}|>\n"

        if not reference:  # no retriever
            messages = [
                {"role": "system", "content": "You are a professional radiologist."},
                {"role": "user", "content": placeholder + "Please generate a report based on the Chest X-ray image. Just report, not adding further information."}
            ]
        else:
            reference_report = line["reference_reports"]
            if not isinstance(reference_report, list):
                topk = 1
                reference_report = [reference_report]
                formatted_reference_report = reference_report[0]
            else:
                topk = len(reference_report)
                formatted_reference_report = ""
                for i in range(topk):
                    formatted_reference_report += f"{i + 1}. {reference_report[i]} "
            messages = [
                {"role": "system", "content": f"You are a professional radiologist. You are provided with an X-ray image and {topk} reference report(s): " + formatted_reference_report + "\n"},
                {"role": "user", "content": placeholder + "\nPlease generate a report based on the image. It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, but should only be used for reference and comparison. Please only include the content of the report in your response."}
            ]

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 2048,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "answer": response,
            "gt_answer": gt_answer,
            "image": image_file,
            "image_id": image_file,
            "answer_id": ans_id,
            "model_id": model_id,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phi-3.5 Vision report generation.")
    parser.add_argument("--model_id", type=str, default="Lexius/Phi-3.5-vision-instruct", help="Model ID or path")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--question_file", type=str, required=True, help="JSONL file with questions")
    parser.add_argument("--answer_file", type=str, default="answer_file.jsonl", help="Output file for answers")
    parser.add_argument("--reference", action="store_true", help="Use reference reports if available")

    args = parser.parse_args()

    main(args.model_id, args.image_folder, args.question_file, args.answer_file, args.reference)
