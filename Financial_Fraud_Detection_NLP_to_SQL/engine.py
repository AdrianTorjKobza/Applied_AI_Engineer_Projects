# Prompt engineering and the multi-tenant mapping logic.

import json
from sqlglot import transpile, parse_one, exp

class FraudSqlEngine:
    def __init__(self):
        # In a real RAG, these would be fetched from a Vector DB based on Tenant ID.
        self.schema_context = """
        Table: transactions
        Columns:
        - tx_id (INTEGER)
        - amount (DECIMAL)
        - currency (VARCHAR)
        - country (VARCHAR)
        - timestamp (DATETIME)
        """

    def build_prompt(self, natural_query):
        return f"""
        [INST] <<SYS>>
        You are a SQL expert for a bank. Convert the user's request into a valid PostgreSQL query.
        Return ONLY a JSON object with two keys: "sql" and "explanation".
        
        Context Schema:
        {self.schema_context}
        <</SYS>>
        User Request: {natural_query} [/INST]
        """

    def validate_sql(self, sql_query):
        """Basic syntax validation using sqlglot."""
        try:
            # Check if it's valid SQL and transpiled to Postgres dialect.
            parsed = parse_one(sql_query, read="postgres")
            return True, sql_query
        except Exception as e:
            return False, str(e)