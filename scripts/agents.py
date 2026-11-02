from sentence_transformers import SentenceTransformer
from smolagents import TransformersModel, CodeAgent, Tool, tool, RunResult
import json
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from simple_ddl_parser import parse_from_file
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
    description = "Allows for queries to be executed on a database specified by db_id. Returns the results of the query."
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
            raise ValueError(f"Query execution failed. Make sure the query is in correct SQL format. Full trace: {e}")


class GetSchemaTool(Tool):
    name = "get_schema"
    description = "Allows for viewing the SQL schema of a database specified by db_id."
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


class TableSelectorTool(Tool):
    name = "select_tables"
    description = "Given a database ID, a threshold value and a natural-language question, returns schema information for only the tables sufficiently relevant to the question"
    inputs = {
        "db_id": {
            "type": "string",
            "description": "The database ID."
        },
        "threshold": {
            "type": "number",
            "description": "The threshold value for determining whether a table is relevant. Should be between 0 and 1. 0.2 is the recommended starting value."
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
        for statement in statements:
            if statement.startswith("CREATE TABLE"):
                table_name = statement.split('"')[1]
                table_embedding = self.encoder.encode(statement)
                tables.append({"table_name": table_name, "schema": statement, "embedding": table_embedding})
        self.dbs[db_id] = tables

    # def _setup_db_info(self, db_id: str):
    #     full = parse_from_file(f'{self.db_root}/{db_id}/schema.sql')
    #
    #     table_keys = {"table_name", "primary_key", "columns"}
    #     column_keys = {"name", "type", "references", "unique"}
    #     references_keys = {"table", "column"}
    #
    #     tables = []
    #     for table_dict in full:
    #         table = {k: table_dict[k] for k in table_keys if k in table_dict}
    #         columns = []
    #         for col_dict in table["columns"]:
    #             column = {k: col_dict[k] for k in column_keys if k in col_dict}
    #             if column["references"] is not None:
    #                 column["references"] = {k: column["references"][k] for k in references_keys if
    #                                         k in column["references"]}
    #             columns.append(column)
    #         table["columns"] = columns
    #         table_embedding = self.encoder.encode(str(table))
    #         table_description = f"Table: {table['table_name']}\n"
    #         table_description += "Columns:\n" + "\n".join([f"  - {col['name']}: {col['type']}" for col in columns])
    #         tables.append({"table_name": table["table_name"], "schema": table_description, "embedding": table_embedding})
    #     self.dbs[db_id] = tables

    def _get_db_info(self, db_id: str):
        if db_id not in self.dbs.keys():
            self._setup_db_info(db_id)
        return self.dbs[db_id]

    def forward(self, db_id: str, threshold: float, question: str) -> list[str]:
        question_embedding = self.encoder.encode(question)
        tables = self._get_db_info(db_id)

        similar_tables = []
        for table in tables:
            if cosine_similarity(table["embedding"].reshape(1, -1), question_embedding.reshape(1, -1)) >= threshold:
                similar_tables.append(table["schema"])

        return similar_tables


class ExampleRetrieverTool:
    name = "example_retriever"
    description = "For a given question, finds 3 similar examples and returns the example question, similarity score (1 is best) and example query, in that order, for each."
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
        example_embeddings = list(self.embeddings.values())
        similarities = cosine_similarity(q_embedding, example_embeddings)
        top_k_embeddings = np.top_k(similarities, k=self.k, axis=1)
        top_examples = []
        for val, idx in top_k_embeddings:
            ex_question = self.examples[idx]["question"]
            ex_query = self.examples[idx]["query"]
            top_examples.append(f"Question: {ex_question}\nCosine Similarity: {val}\nQuery: {ex_query}")
        return top_examples


class AgentHarness:
    def __init__(
            self,
            model_id: str,
            gen_config: dict,
            tools: list[Tool],
    ):
        self.engine = TransformersModel(
            model_id=model_id,
            device_map="balanced",
            trust_remote_code=False,
            **gen_config
        )
        self.tools = tools
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.engine
        )

    def run(self, question: str, db_id: str, return_full_result: bool = False) -> RunResult:
        result = self.agent.run(
            task=question,
            additional_args={
                "db_id": db_id
            },
            return_full_result=return_full_result
        )
        return result



