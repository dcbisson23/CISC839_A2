from transformers import AutoTokenizer, AutoModelForCausalLM
from smolagents import CodeAgent, TransformersModel, tool
import gc
import torch
import json
from tqdm import tqdm

def load_model(model_dict:dict) -> AutoModelForCausalLM:
    model_name = model_dict.get("model")
    gguf = model_dict.get("gguf")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced")
    return model


def load_tokenizer(model_dict:dict) -> AutoTokenizer:
    model_name = model_dict.get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="balanced")
    return tokenizer


def load_gen_config(gen_config_path:str) -> dict:
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    return gen_config


def unload_model(model):
    # It annoys me that this doesn't work.
    # .ipynb files have weird VRAM management apparently? Uncertain.
    del model
    gc.collect()
    torch.cuda.empty_cache()


def get_questions(questions_path: str, sample_rate: float, offset: int=0):
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)

    db_sample_accum = {}
    total_samples = 0
    questions = []
    adding_offset_flag = False
    offset_counter = 0
    for idx, item in tqdm(enumerate(questions_data), desc="Generating prompts"):

        db_id = item['db_id']
        # We only want a # of samples for each db proportional to the # of questions available to the db.
        # So, take the first question for each db, then only take questions when enough questions have gone by.
        # Ex:
        # - for sample_rate = 1, take every question.
        # - for sample_rate = 0.5, take every other question.
        skip_flag = True
        if db_sample_accum.get(db_id) is None:
            db_sample_accum[db_id] = 0.0
            adding_offset_flag = True
        else:
            db_sample_accum[db_id] += sample_rate
            if db_sample_accum[db_id] >= 1.0:
                db_sample_accum[db_id] -= 1.0
                skip_flag = False

        # This just offsets the sampling so we can define multiple distinct subsets.
        if adding_offset_flag:
            if offset_counter >= offset:
                offset_counter = 0
                adding_offset_flag = False
                skip_flag = False
            else:
                offset_counter += 1

        if skip_flag:
            continue
        question = item['question']

        questions.append({"id": idx, "db_id": db_id, "question": question})
    return questions


def get_data_from_response(response: str) -> dict:
    response_data = {}
    res_split = response
    # Get any thinking blocks first.
    res_split = res_split.split("</think>")
    thinking_blocks = res_split
    res_split = [thinking_blocks.pop(-1)]
    if len(thinking_blocks) > 0:
        for block in thinking_blocks:
            # Thinking blocks come before anything else, so this should work.
            block.removeprefix("<think>")
    response_data["reasoning_trace"] = "\n".join(thinking_blocks)
    res_split = res_split[-1].split("```")
    if len(res_split) > 1:
        # We're only going to evaluate on the last SQL segment if multiple exist.
        query = res_split[-2].removeprefix("sql")
    else:
        # If at this point we don't have any sql blocks, we'll assume the response is formatted correctly.
        query = res_split[-1]
    # We need the query to not have any newlines for the evaluation script to work on it.
    # Decided to also remove tabs because tabs spaces and newlines between keywords are treated the same in an SQL query.
    # Just makes it look nicer I think.
    query = query.replace("\n", " ")
    query = query.replace("\t", " ")
    query = " ".join(query.split())
    response_data["prediction"] = query.strip()

    return response_data
