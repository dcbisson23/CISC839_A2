from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm


class LLMPrompter:
    def __init__(self, model:AutoModelForCausalLM, tokenizer:AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = {}

    def load_gen_config(self, gen_config_path:str):
        with open(gen_config_path, 'r') as f:
            self.gen_config = json.load(f)

    def prompt(self, prompt:str) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            **self.gen_config
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

    def run_prompt_pipeline(self, prompts_path:str, output_path:str):
        with open(prompts_path, 'r') as f:
            prompts = f.readlines()
        with open(output_path, "w+") as f_out:
            # We want a list of existing responses to make the pipeline hotstart for incomplete response files.
            existing_outputs = f_out.readlines()
            existing_ids = []
            for output_json in existing_outputs:
                existing_output_record = json.loads(output_json)
                existing_ids.append(existing_output_record["id"])
            output_records = []
            for prompt_json in tqdm(prompts, desc=f"Generating responses"):
                prompt_record = json.loads(prompt_json)
                id = prompt_record["id"]
                db_id = prompt_record["db_id"]
                prompt = prompt_record["prompt"]
                # We don't need to redo any existing responses.
                if id in existing_ids:
                    existing_ids.remove(id)
                    continue

                response = self.prompt(prompt)
                output_record = {
                    "id": id,
                    "db_id": db_id,
                    "response": response,
                }
                output_records.append(output_record)
                f_out.write(json.dumps(output_record) + "\n")
        print(f"Successfully wrote {len(prompts)} responses to {output_path}")
        return output_records
