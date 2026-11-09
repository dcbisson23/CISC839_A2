from transformers import AutoTokenizer, AutoModelForCausalLM
from smolagents import CodeAgent, TransformersModel, tool
import gc
import torch
import json
from tqdm import tqdm
import re

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


TASK_FORMAT = r""".*\{'question': (?:"(.*)"|'(.*)'), 'db_id': '(.*)'}."""


def get_question_from_task(task: str):
    match = re.match(TASK_FORMAT, task, re.DOTALL)
    if match is not None:
        if match.groups()[0] is None:
            question = match.groups()[1]
        else:
            question = match.groups()[0]
        db_id = match.groups()[2]
        return question, db_id
    raise ValueError(f"{task}")


TOOL_OBS_FORMAT = {
    "query_db": r"Query:\n\t(.*)\nValue(s):",
    "get_schema": r"Database ID: (.*)\nSchema:\n",
    "get_relevant_table_data": r"Database ID: (.*)\nQuestion:\n\t(.*)\n",
    "example_retriever": r"Reference Question:\n\t(.*) Similar Examples:\n"
}
ERROR_FORMAT = r"Error executing tool '(.*)' with arguments \{(.*)\}: (.*):"
ACTION_ARGS_FORMAT = r"'(.*)': '(.*)'(?:, )?"


def parse_step_data(observation: str, error_str: str):
    error = None
    if error_str is not None:
        match = re.match(ERROR_FORMAT, error_str)
        if match is not None:
            action = match.groups()[0]
            action_args_str = match.groups()[1]
            action_args = re.findall(ACTION_ARGS_FORMAT, action_args_str)
            action_args = dict(action_args)
            error = match.groups()[2]
            return action, action_args, error
        elif error_str == "Reached max steps.":
            error = "MaxStepsError"
        else:
            error = "Other"
    if observation is not None:
        action = ""
        action_args = []
        for key, value in TOOL_OBS_FORMAT.items():
            match = re.match(value, observation)
            if match is not None:
                action = key
                match key:
                    case "query_db":
                        action_args.append(("query", match.groups()[0]))
                    case "get_schema":
                        action_args.append(("db_id", match.groups()[0]))
                    case "get_relevant_table_data":
                        action_args.append(("db_id", match.groups()[0]))
                        action_args.append(("question", match.groups()[1]))
                    case "example_retriever":
                        action_args.append(("question", match.groups()[0]))
                action_args = dict(action_args)
                return action, action_args, error
        return "final_answer", {"answer": observation}, error
    return None, None, error


def get_trajectory_features(agent_run_json: str, give_full_trajectory: bool = False):
    run_data = json.loads(agent_run_json)
    output = run_data["output"]
    state = run_data["state"]
    steps = run_data["steps"]
    token_usage = run_data["token_usage"]
    timing = run_data["timing"]

    # The features to extract from the trace
    repeated_actions = []
    num_steps = 0

    if state == "max_steps_error":
        # Output is whatever the LLM provides when steps are maxed out, rather than actual output.
        # Easier to set to None.
        output = None

    action_dict = {}
    repeated_actions = {}

    # First step in trajectory is the task.
    trajectory = [steps[0]["task"]]
    question, db_id = get_question_from_task(trajectory[0])

    for step in steps[1:]:
        step_num = step["step"]
        if step_num > 10:
            break
        num_steps += 1
        action, action_args, error = parse_step_data(step["observations"], step["error"])
        if action in action_dict:
            if action_args in action_dict[action]:
                if action not in repeated_actions:
                    repeated_actions[action] = {}
                repeated_actions[action] = action_args
            else:
                action_dict[action].append(action_args)
        else:
            action_dict[action] = [action_args]
        if give_full_trajectory:
            trajectory.append({"step": step,
                               "action": action,
                               "action_args": action_args,
                               "observation": step["observation"],
                               "error": error})
    # No need for duplicates here, but order is important.
    if give_full_trajectory:
        return question, db_id, output, num_steps, repeated_actions, token_usage, timing, trajectory
    return question, db_id, output, num_steps, repeated_actions, token_usage, timing












