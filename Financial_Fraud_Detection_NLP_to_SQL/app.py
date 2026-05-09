import streamlit as st
import pandas as pd
import requests
import json
import re
from engine import FraudSqlEngine
from mock_db import get_mock_db

# Initialize Engine and DB.
engine_logic = FraudSqlEngine()
db_engine = get_mock_db()

st.set_page_config(page_title="FraudGuard NLP", layout="wide")

st.title("🛡️ FraudGuard: Natural Language to SQL")
st.markdown("### PoC: Financial Rule Generation for Analysts")

with st.sidebar:
    st.header("Settings")
    ollama_url = st.text_input("Ollama Endpoint", value="http://localhost:11434/api/generate")
    model_name = st.text_input("Model", value="llama3:8b")
    st.info("Ensure Ollama is running: `ollama run llama3:8b`")

def extract_json(text):
    """Robustly extract JSON from a string that might contain text/markdown."""
    try:
        # Look for a JSON block between markdown markers
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())
        # Fallback to parsing the whole text if no braces found.
        return json.loads(text)
    except Exception:
        return None

query_input = st.text_area("Describe the fraud rule in plain English:", 
                          placeholder="Show me transactions from US with amount greater than 5000")

if st.button("Generate & Execute"):
    if query_input:
        with st.spinner("Processing with Llama 3..."):
            prompt = engine_logic.build_prompt(query_input)
            
            try:
                # API Call to local Ollama.
                response = requests.post(ollama_url, 
                                         json={"model": model_name, "prompt": prompt, "stream": False},
                                         timeout=45)
                response.raise_for_status()
                
                # Check if 'response' key exists in the JSON returned by Ollama.
                resp_json = response.json()

                if 'response' not in resp_json:
                    st.error(f"Unexpected API response format: {resp_json}")
                    st.stop()
                
                raw_llm_text = resp_json['response']
                
                # Extract and Parse JSON.
                data = extract_json(raw_llm_text)
                
                if data and 'sql' in data:
                    generated_sql = data['sql']
                    
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Generated SQL")
                        st.code(generated_sql, language='sql')
                        st.subheader("Logic Explanation")
                        st.write(data.get('explanation', 'No explanation provided.'))
                    
                    # Validation & Execution
                    is_valid, msg = engine_logic.validate_sql(generated_sql)
                    
                    with col2:
                        st.subheader("Execution Results")
                        if is_valid:
                            results = pd.read_sql(generated_sql, db_engine)
                            st.success("Query Validated Successfully")
                            st.dataframe(results)
                        else:
                            st.error(f"SQL Validation Error: {msg}")
                else:
                    st.error("LLM failed to return valid JSON format.")
                    st.text("Raw Response for Debugging:")
                    st.code(raw_llm_text)

            except Exception as e:
                st.error(f"System Error: {str(e)}")