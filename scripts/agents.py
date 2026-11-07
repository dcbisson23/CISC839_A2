from sentence_transformers import SentenceTransformer
from smolagents import TransformersModel, CodeAgent, ToolCallingAgent, Tool, tool, RunResult, ActionStep, ChatMessage, \
    PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
import json
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlparse
import sqlvalidator
from torch import Tensor


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
        return schema


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
        return table_strings


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
        return top_examples


def validate_final_answer(final_answer, memory, agent):
    answer_sql = sqlvalidator.parse(final_answer)
    if answer_sql.is_valid():
        return True
    else:
        raise ValueError("\"answer\" string in JSON blob must be a valid SQL query.")


class CustomAgent(ToolCallingAgent):
    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        # Modified to reduce memory usage.
        # Keeps system prompt, removes agent thoughts in each step but retains action/observation.

        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=True))
        return messages


class AgentHarness:
    def __init__(
            self,
            model_id: str,
            gen_config: dict,
            tools: list[Tool],
            sys_prompt: str
    ):

        self.engine = TransformersModel(
            model_id=model_id,
            device_map="balanced",
            trust_remote_code=False,
            **gen_config
        )
        self.tools = tools
        self.agent = CustomAgent(
            tools=self.tools,
            model=self.engine,
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
