TABLES_PATH = "spider/tables.json"
QUESTIONS_PATH = "spider/dev.json"
GOLD_PATH = "spider/dev_gold.sql"

# This is where model data (generation configs, formatted prompts, responses) are written to/read from.
DATA_ROOT = "data"
GEN_CONFIG_FILENAME = "generation_config.json"
PROMPTS_FILENAME = "formatted_prompts.jsonl"
COT_PROMPTS_FILENAME = "cot_formatted_prompts.jsonl"
RESPONSES_FILENAME = "responses.jsonl"
COT_RESPONSES_FILENAME = "cot_responses.jsonl"
PREDICTIONS_FILENAME = "predictions.txt"
COT_PREDICTIONS_FILENAME = "cot_predictions.txt"
GOLD_FILENAME = "gold.txt"
REASONING_FILENAME = "reasoning.jsonl"

# Code is statically saved in /scripts/.
MODELS = [{"model": "Qwen/Qwen2.5-Coder-3B-Instruct", "can_reason": False, "use_in_q3": True},
          {"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "can_reason": False, "use_in_q3": False},
          {"model": "Qwen/Qwen3-4B-Thinking-2507", "can_reason": True, "use_in_q3": False}]

SAMPLE_RATE = 0.125

# These strings are used in the formatted prompts.
# Put them here because it's useful to have them accessible.
SYS_CONTEXT = ("""You are an expert SQLite developer, tasked with writing a valid SQL query to answer a user question using a given SQL schema.
The user will provide the following information:
    1. The database schema as a standard SQLite DDL script, and
    3. The question to be answered by querying this database.
Your response must adhere to the following guidelines:
    1. (CRITICAL) You must provide ONLY a valid SQLite query to answer the provided question, and nothing else.
    2. Use ONLY the tables and columns provided in the schema.
""")
# This is used in the prompts for Task 3.
COT_CONTEXT = ("""
To create the query, first answer the following questions: 
    1. Without referencing the database, how would you translate the question into an SQLite query?
    
    2. What table columns in the database contain data relevant to the question, and why?
    
    3. How do the columns in (2) map to the query in (1)?

Use the answers to these questions to build the final query.
""")
