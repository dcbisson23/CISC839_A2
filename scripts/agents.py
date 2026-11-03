from sentence_transformers import SentenceTransformer
from smolagents import TransformersModel, ToolCallingAgent, Tool, tool, RunResult, PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
import json
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlparse


@tool
def get_schema(db_path: str) -> str:
    """
    Pulls a database schema from the schema file.
    Args:
         db_path: The path to the SQL schema file.
    """
    with open(db_path, "r") as f:
        schema = f.read()
    return schema


class QueryTool(Tool):
    name = "query_db"
    description = ("Allows for queries to be tested on a database specified by db_id. "
                   "Use this to verify an SQL query works on the database.")
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to be executed on the database."
        },
        "db_id": {
            "type": "string",
            "description": "The ID of the database to be queried."
        }
    }
    output_type = "array"

    def __init__(self, db_root: str):
        super().__init__()
        self.db_root = db_root
        # Make a cache of connections for databases the tool has queried, for ease of access.
        self.connections = {}

    def forward(self, query: str, db_id: str) -> list[str]:
        # Define a read-only connection to a database if none already exists.
        if db_id not in self.connections:
            try:
                self.connections[db_id] = sqlite3.connect(f"file:{self.db_root}/{db_id}/{db_id}.sqlite?mode=ro", uri=True)
            except Exception as e:
                raise ValueError(f"Database does not exist. Please provide a valid database ID. Full trace: {e}")
        cursor = self.connections[db_id].cursor()
        try:
            cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            raise ValueError(f"Query execution failed. Make sure the query is in correct SQL format, "
                             f"and that all table/column references exist in the database by cross-referencing with the database schema. "
                             f"Full trace: {e}")


class GetSchemaTool(Tool):
    name = "get_schema"
    description = "Returns the full SQL schema of a database specified by db_id."
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
        self.schemas = {}

    def forward(self, db_id: str) -> str:
        if db_id not in self.schemas:
            with open(f"{self.db_root}/{db_id}/schema.sql", "r") as f:
                self.schemas[db_id] = f.read()
        schema = self.schemas[db_id]
        return schema


class TableOrdererTool(Tool):
    name = "order_tables"
    description = ("Given a database ID, a threshold value and a natural-language question, "
                   "returns the table schemas in order of most to least relevant to the question."
                   "Prioritize using the most relevant tables when generating the SQL query to answer the question.")
    inputs = {
        "db_id": {
            "type": "string",
            "description": "The database ID."
        },
        "question": {
            "type": "string",
            "description": "The natural language question for the database."
        }
    }
    output_type = "array"

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
        embeddings = []
        for idx, statement in enumerate(statements):
            if statement.startswith("CREATE TABLE"):
                table_name = statement.split('"')[1]
                table_embedding = self.encoder.encode(statement)
                tables.append({"table_name": table_name, "schema": statement})
                embeddings.append(table_embedding)
        self.dbs[db_id] = {"tables": tables, "embeddings": embeddings}

    def _get_db_info(self, db_id: str):
        if db_id not in self.dbs.keys():
            self._setup_db_info(db_id)
        return self.dbs[db_id]

    def forward(self, db_id: str, question: str) -> list[str]:
        question_embedding = np.array(self.encoder.encode(question)).reshape(1, -1)
        db_info = self._get_db_info(db_id)
        tables = db_info["tables"]
        table_embeddings = np.array(db_info["embeddings"])
        similarities = cosine_similarity(question_embedding, table_embeddings)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_tables = []
        for idx in sorted_indices[0]:
            sorted_tables.append(tables[idx]["schema"])

        return sorted_tables


class ExampleRetrieverTool(Tool):
    name = "example_retriever"
    description = ("For a given natural-language question to be converted to an SQL query, "
                   "finds 3 examples of similar questions and SQL queries that answer them. "
                   "For each example, returns the example question, "
                   "an SQL query that answers the question, "
                   "and the similarity score that measures how similar the example question is to the original question (1 is best), "
                   "in that order."
                   "Use this tool to get good examples of how similar questions are converted into SQL queries, "
                   "to help inform generation of an SQL query for the provided question.")
    inputs = {
        "question": {
            "type": "string",
            "description": "The natural-language question, which the tool will use to find similar examples."
        }
    }
    output_type = "array"

    def __init__(self, encoder: SentenceTransformer, examples_path: str):
        super().__init__()
        self.encoder = encoder
        self.examples_path = examples_path
        # Was wanting to make this a parameter, but it's important the agent knows how many examples to expect.
        self.k = 3
        with open(self.examples_path, "r") as f:
            examples_dict = json.load(f)
        self.embeddings = {}
        examples = []
        for idx, example in enumerate(examples_dict):
            question = example["question"]
            embedding = self.encoder.encode(question)
            query = example["query"]
            self.embeddings[idx] = embedding
            examples.append({
                "question": question,
                "query": query
            })
        self.examples = examples

    def forward(self, question: str) -> list[str]:
        q_embedding = self.encoder.encode(question).reshape(1, -1)
        example_embeddings = np.array(list(self.embeddings.values()))
        similarities = cosine_similarity(q_embedding, example_embeddings)
        top_k_indices = np.argpartition(similarities, -self.k, axis=1)[0][-self.k:]
        top_examples = []
        for idx in top_k_indices.tolist():
            ex_question = self.examples[idx]["question"]
            ex_query = self.examples[idx]["query"]
            ex_val = similarities[0][idx]
            top_examples.append(f"Question: {ex_question}\nQuery: {ex_query}\nCosine Similarity: {ex_val}")
        return top_examples


class AgentHarness:
    def __init__(
            self,
            model_id: str,
            gen_config: dict,
            tools: list[Tool]
    ):
        self.engine = TransformersModel(
            model_id=model_id,
            device_map="balanced",
            trust_remote_code=False,
            **gen_config
        )
        self.tools = tools
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=self.engine,
            planning_interval=3
        )

    def run(self,
            question: str,
            db_id: str,
            task: str,
            stream_outputs: bool = False,
            return_full_result: bool = False) -> RunResult:
        result = self.agent.run(
            task=task,
            stream=stream_outputs,
            additional_args={
                "question": question,
                "db_id": db_id
            },
            return_full_result=return_full_result
        )
        return result



