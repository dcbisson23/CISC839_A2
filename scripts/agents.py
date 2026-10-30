from smolagents import TransformersModel, CodeAgent, Tool, tool, RunResult
import json
import sqlite3

@tool
def get_schema(db_id: str) -> str:
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
    description = "Allows for queries to be executed on a database specified by db_id. Returns the result of the query."
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
    output_type = "string"

    def __init__(self, db_root: str):
        super().__init__()
        self.db_root = db_root
        # Make a cache of connections for databases the tool has queried, for ease of access.
        self.connections = {}

    def forward(self, query: str, db_id: str) -> str:
        # Define a read-only connection to a database if none already exists.
        if db_id not in self.connections:
            self.connections[db_id] = sqlite3.connect(f"{self.db_root}/{db_id}/{db_id}.sqlite?mode=ro", uri=True)
        cursor = self.connections[db_id].cursor()
        try:
            cursor.execute(query)
            return str(cursor.fetchall())
        except Exception as e:
            return f"Query execution failed: {e}"


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


class AgentHarness:
    def __init__(
            self,
            model_id: str,
            gen_config: dict,
            tools: list[Tool],
            db_root: str
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

        self.db_root = db_root

    def run(self, question: str, db_id: str, return_full_result: bool = False) -> RunResult:
        result = self.agent.run(
            task=question,
            additional_args={
                "db_id": db_id
            },
            return_full_result=return_full_result
        )
        return result



