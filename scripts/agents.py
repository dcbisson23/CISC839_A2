import re
import uuid

from sentence_transformers import SentenceTransformer
from smolagents import TransformersModel, CodeAgent, ToolCallingAgent, Tool, tool, RunResult, ActionStep, ChatMessage, \
    PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate, MessageRole, \
    TokenUsage
from smolagents.models import remove_content_after_stop_sequences, ChatMessageToolCall, parse_json_if_needed, \
    ChatMessageToolCallFunction
from smolagents.utils import Retrying, AgentParsingError
import json
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlparse
import sqlvalidator
import torch
import gc
from torch import Tensor
import json_repair


class QueryTool(Tool):
    name = "query_db"
    description = ("Attempts to run an SQL query on the specified database.\n"
                   "Returns the query and the value(s) the query returns when executed.")
    inputs = {
        "query": {
            "type": "string",
            "description": "The SQL query."
        },
        "db_id": {
            "type": "string",
            "description": "The ID of the database to be queried."
        }
    }
    output_type = "string"

    def __init__(self, db_root: str):
        super().__init__()
        self.db_root = db_root
        # Make a cache of connections for databases the tool has queried, for ease of access.
        # EDIT: No longer a full cache, only stores last connection called. Hopefully reduces memory usage.
        self.connections = {}

    def forward(self, query: str, db_id: str) -> str:
        # Define a read-only connection to a database if none already exists.
        if db_id not in self.connections:
            try:
                new_connection = sqlite3.connect(f"file:{self.db_root}/{db_id}/{db_id}.sqlite?mode=ro", uri=True)
                self.connections = {db_id: new_connection}
            except Exception as e:
                raise ValueError(f"Database with specified ID does not exist.\nFull trace:\n{e}")
        cursor = self.connections[db_id].cursor()
        try:
            cursor.execute(query)
            output = f"Query:\n\t{query}\n"
            return output+"Value(s):\n\t"+"\n\t".join([str(item) for item in cursor.fetchall()])
        except Exception as e:
            raise ValueError(f"Query execution failed.\n"
                             f"Make sure the query is in correct SQL format.\n"
                             f"Full trace:\n{e}")


class GetSchemaTool(Tool):
    name = "get_schema"
    description = "Returns the SQL schema of a database specified by db_id."
    inputs = {
        "db_id": {
            "type": "string",
            "description": "The ID of the database."
        }
    }
    output_type = "string"

    def __init__(self, db_root: str):
        super().__init__()
        self.db_root = db_root
        # Make a cache for schemas the tool has used, for ease of access.
        # EDIT: Now stores only the most recently-called schema, to save memory.
        self.schemas = {}

    def forward(self, db_id: str) -> str:
        if db_id not in self.schemas:
            with open(f"{self.db_root}/{db_id}/schema.sql", "r") as f:
                db_schema = f.read()
            statements = sqlparse.split(db_schema)
            self.schemas = {db_id: "\n".join(statements)}

        schema = self.schemas[db_id]
        return f"Database ID: {db_id}\nSchema:\n{schema}"


class TableOrdererTool(Tool):
    name = "get_relevant_table_data"
    description = ("Given a database ID and a natural-language question, "
                   "scores each table and its columns by how relevant they are to the question.\n"
                   "Scores are between 0 and 1, where 1 is best.")
    inputs = {
        "db_id": {
            "type": "string",
            "description": "The database ID."
        },
        "question": {
            "type": "string",
            "description": "The natural-language question."
        },
    }
    output_type = "string"

    def __init__(self, db_root: str, encoder: SentenceTransformer):
        super().__init__()
        self.db_root = db_root
        self.encoder = encoder

        self.dbs = {}

    def _setup_db_info(self, db_id: str):
        with open(f"{self.db_root}/{db_id}/schema.sql", "r") as f:
            db_schema = f.read()
        statements = sqlparse.split(db_schema)
        tables = []
        table_strings = []
        for idx, statement in enumerate(statements):
            if statement.startswith("CREATE TABLE"):
                table_name = statement.split('"')[1]
                column_schema = statement[statement.index("(")+1:statement.rindex(");")]
                column_lines = [col.replace("\n", "") for col in column_schema.split(",\n")]
                column_names = []
                # Filter out primary/foreign key references (we just want columns).
                for line in column_lines:
                    if line.find("PRIMARY KEY") == -1 and line.find("FOREIGN KEY") == -1:
                        column_name = line.split(" ")[0]
                        column_names.append("\""+table_name+"."+column_name[1:])
                column_embeddings = self.encoder.encode(column_names, normalize_embeddings=True)
                columns = "\n\t".join(column_names)
                table_string = f"Table: {table_name}\nColumns:\n\t{columns}"
                table_strings.append(table_string)
                tables.append({"table_name": table_name,
                               "column_names": column_names,
                               "column_embeddings": column_embeddings})
        table_embeddings = self.encoder.encode(table_strings, normalize_embeddings=True)
        self.dbs = {db_id: {"tables": tables, "table_embeddings": table_embeddings}}

    def _get_db_info(self, db_id: str):
        if db_id not in self.dbs.keys():
            self._setup_db_info(db_id)
        return self.dbs[db_id]

    def forward(self, db_id: str, question: str) -> str:
        db_info = self._get_db_info(db_id)
        question_embedding = self.encoder.encode([question], normalize_embeddings=True)
        table_embeddings = db_info["table_embeddings"]
        table_scores = np.array(self.encoder.similarity(question_embedding, table_embeddings)).flatten()
        table_order = np.argsort(table_scores)[::-1].flatten()
        table_strings = []
        for t_idx, table_dict in enumerate(db_info["tables"]):
            table_score = table_scores[t_idx]
            column_names = table_dict["column_names"]
            column_embeddings = table_dict["column_embeddings"]
            column_scores = np.array(self.encoder.similarity(question_embedding, column_embeddings).flatten())
            column_order = np.argsort(column_scores)[::-1].flatten()
            column_strings = []
            for c_idx, score in enumerate(column_scores.flatten()):
                column_string = f"{column_names[c_idx]}: {score:.4f}"
                column_strings.append(column_string)
            column_strings = list(np.array(column_strings)[column_order])
            column_strings = "\n\t\t".join(column_strings)
            table_string = (f"Table {table_dict['table_name']}:\n"
                            f"\tTable score: {table_score:.4f}\n"
                            f"\tColumn scores:\n"
                            f"\t\t{column_strings}")
            table_strings.append(table_string)
        table_strings = list(np.array(table_strings)[table_order])
        table_strings = "\n\n".join(table_strings)
        return (f"Database ID: {db_id}\n"
                f"Question:\n\t{question}\n"
                f"{table_strings}")

class ExampleRetrieverTool(Tool):
    name = "example_retriever"
    description = ("For a given natural-language question to be converted to an SQL query, "
                   "returns 3 examples of similar questions and SQL queries that answer them.\n"
                   "Also scores how similar each example is to the original question (1 is best).\n"
                   "Use this tool to get good examples of queries for similar problems.")
    inputs = {
        "question": {
            "type": "string",
            "description": "The natural-language question."
        }
    }
    output_type = "string"

    def __init__(self, encoder: SentenceTransformer, examples_path: str):
        super().__init__()
        self.encoder = encoder
        self.examples_path = examples_path
        # Was wanting to make this a parameter, but it's important the agent knows how many examples to expect.
        self.k = 3
        with open(self.examples_path, "r") as f:
            examples_dict = json.load(f)
        self.questions = [example["question"] for example in examples_dict]
        self.embeddings = self.encoder.encode(self.questions)
        self.queries = [example["query"] for example in examples_dict]

    def forward(self, question: str) -> str:
        q_embedding = self.encoder.encode(question)
        similarities = self.encoder.similarity(q_embedding, self.embeddings).flatten()
        top_k_indices = np.argpartition(similarities, -self.k)[-self.k:]
        top_examples = []
        for idx in top_k_indices.tolist():
            ex_question = self.questions[idx]
            ex_query = self.queries[idx]
            ex_val = similarities[idx]
            top_examples.append(f"Example Question: {ex_question}\n"
                                f"Query: {ex_query}\n"
                                f"Score: {ex_val}")
        top_examples = "\n\n".join(top_examples)
        return f"Reference Question:\n\t{question} Similar Examples:\n{top_examples}"


def validate_final_answer(final_answer, memory, agent):
    answer_sql = sqlvalidator.parse(final_answer)
    if answer_sql.is_valid():
        return True
    else:
        raise ValueError("\"answer\" string in JSON blob must be a valid SQL query.")


class CustomAgent(ToolCallingAgent):
    def __init__(self, max_retries: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries=max_retries

    # def write_memory_to_messages(
    #     self,
    #     summary_mode: bool = False,
    # ) -> list[ChatMessage]:
    #     # Modified to reduce memory usage.
    #     # Keeps system prompt, removes agent thoughts in each step but retains action/observation.
    #
    #     messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
    #     for memory_step in self.memory.steps:
    #         messages.extend(memory_step.to_messages(summary_mode=True))
    #     return messages

    def _step_stream(self, memory_step: ActionStep):
        # Basically wrapped _step_stream to allow for retrying if the model fails to produce a tool call.
        if self.max_retries <= 0:
            yield from super()._step_stream(memory_step)
        else:
            for i in range(0, self.max_retries+2):
                step_stream = super()._step_stream(memory_step)
                step_list = []
                try:
                    for item in step_stream:
                        step_list.append(item)
                        del item
                    for item in step_list:
                        yield item
                        del item
                    break
                except Exception as e:
                    if isinstance(e, AgentParsingError) and i < self.max_retries:
                        if self.logger is not None:
                            self.logger.log(f"Regenerating (attempt {i+1} of {self.max_retries})...")
                    else:
                        for item in step_list:
                            yield item
                            del item
                        raise
                del step_stream
                del step_list
                gc.collect()
                torch.cuda.empty_cache()


# Had to change the type hint and that's it.
def parse_json_blob(json_blob: str) -> tuple[dict[str, str|dict], str]:
    "Extracts the JSON blob from the input and returns the JSON data and the rest of the input."
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_str = json_blob[first_accolade_index : last_accolade_index + 1]
        json_str = json_repair.repair_json(json_str)
        json_data = json.loads(json_str, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    alt_keys = ["id", "type", "function"]
    not_alt_keys = False
    for key in alt_keys:
        if key not in tool_call_dictionary.keys():
            not_alt_keys = True
            break
    if not not_alt_keys:
        tool_id = tool_call_dictionary["id"]
        tool_type = tool_call_dictionary["type"]
        tool_func = tool_call_dictionary["function"]
        return ChatMessageToolCall(
            id=tool_id,
            type=tool_type,
            function=ChatMessageToolCallFunction(name=tool_func["name"], arguments=tool_func.get("arguments", None))
        )
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Tool call needs to have a key '{tool_name_key}'. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )

class CustomModel(TransformersModel):
    # Straight from TransformersModel class, but removed stop sequences.

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Sometimes APIs do not return the tool call as a specific object, so we need to parse it."""
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message
    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if stop_sequences is not None:
            # output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
            pass
        del generation_kwargs["inputs"]
        del out
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )



class AgentHarness:
    def __init__(
            self,
            model_id: str,
            gen_config: dict,
            tools: list[Tool],
            sys_prompt: str,
            verbosity: int = 1
    ):

        self.engine = CustomModel(
            model_id=model_id,
            device_map="balanced",
            trust_remote_code=False,
            **gen_config
        )
        self.tools = tools
        self.agent = CustomAgent(
            tools=self.tools,
            model=self.engine,
            verbosity_level=verbosity,
            final_answer_checks=[validate_final_answer],
            prompt_templates=PromptTemplates(
                system_prompt=sys_prompt,
                planning=PlanningPromptTemplate(
                    initial_plan="",
                    update_plan_pre_messages="",
                    update_plan_post_messages="",
                ),
                managed_agent=ManagedAgentPromptTemplate(task="", report=""),
                final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
            )
        )

    def run(self,
            question: str,
            db_id: str,
            task: str,
            max_steps: int,
            return_full_result: bool = False) -> RunResult:
        result = self.agent.run(
            task=task,
            max_steps=max_steps,
            additional_args={
                "question": question,
                "db_id": db_id
            },
            return_full_result=return_full_result
        )
        return result
