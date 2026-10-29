import json
from tqdm import tqdm
from transformers import AutoTokenizer


class SpiderPromptBuilder:

    def __init__(self, tables_path: str, questions_path: str, sys_context: str):
        """
        Initializes the builder by loading the Spider dataset files.
        """
        with open(tables_path, 'r', encoding='utf-8') as f:
            self.tables_data = json.load(f)

        with open(questions_path, 'r', encoding='utf-8') as f:
            self.questions_data = json.load(f)

        # Create a lookup dictionary for quick schema retrieval by db_id
        self.schema_lookup = {db['db_id']: db for db in self.tables_data}

        self.sys_context = sys_context

    def _format_schema_to_ddl(self, db_id: str) -> str:
        """
        Converts Spider's index-based schema into SQL CREATE TABLE statements.
        """
        db_info = self.schema_lookup[db_id]
        tables = db_info['table_names_original']
        columns = db_info['column_names_original']
        col_types = db_info['column_types']
        pks = set(db_info['primary_keys'])
        fks= {fk[0]: fk[1] for fk in db_info['foreign_keys']}

        # Map columns to their respective tables
        table_columns = {i: [] for i in range(len(tables))}
        for idx, (tbl_idx, col_name) in enumerate(columns):
            if tbl_idx == -1:  # Skip the '*' column
                continue

            # Map Spider's generic types to standard SQL types
            sql_type = {
                'text': 'TEXT',
                'number': 'NUMERIC',
                'time': 'TEXT',  # SQLite handles time as text usually
                'boolean': 'BOOLEAN',
                'others': 'BLOB'
            }.get(col_types[idx], 'TEXT')

            # The line in SQLite defining each column changes if it's a primary or foreign key.
            # Need to adjust accordingly.
            if idx in pks:
                table_columns[tbl_idx].append(f"  {col_name} {sql_type} PRIMARY KEY")
            elif idx in fks.keys():
                ref_idx = fks[idx]
                ref_table = tables[columns[ref_idx][0]]
                ref_col_name = columns[ref_idx][1]
                table_columns[tbl_idx].append(f"  FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col_name})")
            else:
                table_columns[tbl_idx].append(f"  {col_name} {sql_type}")

        # Build the CREATE TABLE statements
        ddl_statements = []
        for i, tbl_name in enumerate(tables):
            cols = ",\n".join(table_columns[i])
            ddl_statements.append(f"CREATE TABLE {tbl_name} (\n{cols}\n);")
        ddl_statements =  "\n".join(ddl_statements)
        schema_str = f"""```sql\n{ddl_statements}\n```"""

        return schema_str

    def build_chat_template(self, question: str, db_id: str) -> list[dict[str, str]]:
        """
        Constructs the chat template to be applied when prompting the LLM.
        """
        schema_str = self._format_schema_to_ddl(db_id)
        system_context = self.sys_context

        user_question = f"""DATABASE SCHEMA:\n{schema_str}\nQUESTION: \"{question}\"\n"""
        chat_template = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_question},
        ]
        return chat_template

    def generate_prompts(self, output_path: str,
                         tokenizer: AutoTokenizer,
                         sample_rate: float = 1.0,
                         offset: int = 0):
        """
        Iterates through dev.json and generates a JSONL file of prompts and ground truth.
        """
        with open(output_path, 'w', encoding='utf-8') as f_out:
            db_sample_accum = {}
            total_samples = 0
            records = []
            adding_offset_flag = False
            offset_counter = 0
            for i, item in tqdm(enumerate(self.questions_data), desc="Generating prompts"):

                db_id = item['db_id']
                question = item['question']
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

                chat_template = self.build_chat_template(question, db_id)

                prompt = tokenizer.apply_chat_template(
                    chat_template,
                    tokenize=False,
                    add_generation_prompt=True
                )
                record = {
                    "id": item.get("question_id", i),
                    "db_id": db_id,
                    "prompt": prompt,
                }
                records.append(record)
                f_out.write(json.dumps(record) + "\n")
                total_samples += 1

        print(f"Successfully generated {total_samples} prompts to {output_path}")
        return records