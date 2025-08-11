
import os
import io
import streamlit as st
import pandas as pd
import re 
import warnings
import logging
from sqlalchemy import text
from openai import APIConnectionError
import time

from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import AzureChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda

from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

import urllib
from sqlalchemy import create_engine, inspect

########################################### Chatbot #####################################################

# Must be the first Streamlit command in the script.
st.set_page_config(
    page_title="SQL Assistant",  # The title in the browser tab
    page_icon="🧠",             # The icon in the browser tab
    layout="wide"                # Optional: Use wide layout
)

warnings.filterwarnings("ignore")

# Initialize session state variables BEFORE page UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Welcome to the SQL Assistant!")
    ]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "page" not in st.session_state:
    st.session_state.page = "💬 SQL Assistant Chatbot" # Default page

# For storing the last generated SQL query
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# For storing the last response object from the LLM, which includes the result and dataframe
if "last_response" not in st.session_state:
    st.session_state["last_response"] = {} # Initialize as an empty dict

# For storing chat messages (Even though not explicitly in the sidebar, a chatbot needs this)
if "messages" not in st.session_state:
    st.session_state.messages = []

### Load Environment
load_dotenv()

# Cache the Azure OpenAI client
@st.cache_resource
def get_azure_openai_client():
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")  # Required for LangChain
    os.environ["OPENAI_API_TYPE"] = "azure"
    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        request_timeout=840  
    )

@st.cache_resource ##### Function to enable column filtering: manipulates the get_info output
def filter_columns(schema_text: str,
                   table_name: str,
                   allowed_columns: list[str]) -> str:

    allowed_lower = {c.lower() for c in allowed_columns}
    pieces  = re.split(r"(?i)(?=CREATE\s+TABLE)", schema_text)  # keep delimiter
    rebuilt = []
    found   = False

    for chunk in pieces:
        if not chunk.strip():
            continue

        # ── identify table ---------------------------------------------------
        first_line = chunk.splitlines()[0]
        ident = re.sub(r"(?i)^CREATE\s+TABLE\s+", "", first_line)
        ident = re.sub(r'[\[\]`"]', "", ident).split("(")[0].split(".")[-1].strip()

        if ident.lower() != table_name.lower():
            rebuilt.append(chunk)
            continue

        found = True

        # ── locate column list ----------------------------------------------
        pos_open = chunk.find("(")
        if pos_open == -1:
            rebuilt.append(chunk); continue

        depth, pos_close = 0, None
        for i in range(pos_open, len(chunk)):
            if   chunk[i] == "(": depth += 1
            elif chunk[i] == ")":
                depth -= 1
                if depth == 0:
                    pos_close = i
                    break
        if pos_close is None:
            rebuilt.append(chunk); continue

        head        = chunk[:pos_open]
        col_block   = chunk[pos_open+1:pos_close]
        after_paren = chunk[pos_close+1:]

        # ── keep only wanted columns ----------------------------------------
        lines = re.split(r",\s*\n", col_block, flags=re.DOTALL)
        kept  = [
            ln.strip() for ln in lines
            if ln.split()[0].strip('[]"`').lower() in allowed_lower
        ]
        if not kept:
            rebuilt.append(chunk); continue

        # strip anything from '/*' onwards (sample rows)
        after_paren_no_comment = after_paren.split("/*", 1)[0]

        new_chunk = (
            f"{head}(\n  "
            + ",\n  ".join(kept)
            + "\n)"
            + after_paren_no_comment
        )
        rebuilt.append(new_chunk)

    if not found:
        print(f"[filter_columns] Table '{table_name}' not found.")
        return schema_text

    return "".join(rebuilt)

@st.cache_resource ##### Function to enable column filtering: manipulates the get_info output for multipled cols
def filter_columns_multi(schema_text: str,
                         tables: list[str],
                         allowed_columns):

    # ── build {table → set(columns)} ----------------------------------------
    if all(isinstance(c, str) for c in allowed_columns):
        filters = {
            tbl.lower().split(".")[-1]: {c.lower() for c in allowed_columns}
            for tbl in tables
        }
    else:
        if len(tables) != len(allowed_columns):
            raise ValueError("tables and allowed_columns must be the same length")
        filters = {
            tbl.lower().split(".")[-1]: {c.lower() for c in cols}
            for tbl, cols in zip(tables, allowed_columns)
        }

    pieces  = re.split(r"(?i)(?=CREATE\s+TABLE)", schema_text)  # keep delimiter
    rebuilt = []

    for chunk in pieces:
        if not chunk.strip():
            continue

        # ── identify table ---------------------------------------------------
        first_line = next((ln for ln in chunk.splitlines() if ln.strip()), "")
        ident      = re.sub(r"(?i)^CREATE\s+TABLE\s+", "", first_line)
        ident      = re.sub(r'[\[\]`"]', "", ident).split("(")[0].split(".")[-1].strip()
        key        = ident.lower()

        if key not in filters:
            rebuilt.append(chunk)
            continue

        allowed = filters[key]

        # ── locate column list ----------------------------------------------
        pos_open = chunk.find("(")
        if pos_open == -1:
            rebuilt.append(chunk); continue

        depth, pos_close = 0, None
        for i in range(pos_open, len(chunk)):
            if   chunk[i] == "(": depth += 1
            elif chunk[i] == ")":
                depth -= 1
                if depth == 0:
                    pos_close = i
                    break
        if pos_close is None:
            rebuilt.append(chunk); continue

        head        = chunk[:pos_open]
        col_block   = chunk[pos_open+1:pos_close]
        tail        = chunk[pos_close+1:]

        # ── filter column definitions ---------------------------------------
        col_lines = re.split(r",\s*\n", col_block, flags=re.DOTALL)
        kept_cols = [
            ln.strip() for ln in col_lines
            if ln.split()[0].strip('[]"`').lower() in allowed
        ]
        if not kept_cols:
            rebuilt.append(chunk); continue

        ddl_part = f"{head}(\n  " + ",\n  ".join(kept_cols) + "\n)"

        # ── drop sample rows: cut everything from the first /* onwards -------
        tail_no_comment = tail.split("/*", 1)[0]

        rebuilt.append(ddl_part + tail_no_comment)

    return "".join(rebuilt)

# Cache the database connection and SQLDatabase object
@st.cache_resource
def get_sql_database():
    
    # Retrieve the credentials from environment variables
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    username = os.environ.get("SQL_USERNAME")
    password = os.environ.get("SQL_PASSWORD")

    # Build the connection string
    params = urllib.parse.quote_plus(
        f"Driver={driver};"
        f"Server={server};"
        f"Database={database};"
        f"UID={username};"  # Use UID for username
        f"PWD={password};"  # Use PWD for password
    )

    db_uri = f"mssql+pyodbc:///?odbc_connect={params}"
    engine = create_engine(db_uri)

################ Now the database setup starts ###########################
## Since we might not want to use ALL tables and cols, we select a subset 
## Please adjust all table and column lists. also change the schemas in the SQLDatabase function

    # Use table lists
    sales_tables = ["YOUR_TABLE_NAME", "YOUR_TABLE_NAME_2"]
    product_tables = ["YOUR_TABLE_NAME", "YOUR_TABLE_NAME_2"]

    ######## Create databases 
    ## Sales Data

    sales_cols = ["Col1", "Col2"]


    db_sales = SQLDatabase(
        engine=engine,
        include_tables=sales_tables,
        schema="sales",
        sample_rows_in_table_info=2
    )

    filtered_sales = filter_columns_multi(
        schema_text=db_sales.get_table_info(),
        tables=sales_tables,
        allowed_columns=sales_cols
    )

    db_sales.get_table_info = lambda *_, **__: filtered_sales

    ## Product Data
    db_prod = SQLDatabase(
        engine=engine,
        include_tables=product_tables,
        schema="prod",
        sample_rows_in_table_info=2)
    
        ##  Filter cols
    product_cols = ["Product_Col1", "Product_Col2", "Product_Col3"]

    filtered_prod= filter_columns(
        schema_text = db_prod.get_table_info(),
        table_name = "product_tables",                 
        allowed_columns = product_cols)     # keep only these Columns

    db_prod.get_table_info = lambda *_, **__: filtered_prod

    ############# Now combine into one dummy db 

    combined_table_info  = "\n\n".join(
        [db_sales.get_table_info(), db_prod.get_table_info()]
    )

    combined_table_names = db_sales.get_table_names() + db_prod.get_table_names()

    # make them unique and preserve order

    seen = set()
    combined_table_names = [t for t in combined_table_names if not (t in seen or seen.add(t))]

    # Create a SQLDatabase instance
    db = SQLDatabase(
        engine=engine,
        # pass an *empty* MetaData so LangChain won’t reflect anything else
        sample_rows_in_table_info=0,
        lazy_table_reflection=False,
    )

    # override just two methods that the LLM / agent relies on
    db.get_table_info  = lambda *_, **__: combined_table_info
    db.get_table_names = lambda *_, **__: combined_table_names
    
    return engine, db

sql_engine, db = get_sql_database()

########################### Now we build the database to get the columns & schema ######################

## Function to extract the query filters 
@st.cache_resource
def get_sql_filters(query: str) -> list[str]:
    # Normalize the query to lowercase for consistent searching of keywords
    lower_query = query.lower()
    
    # 1. Find the start of the main WHERE clause
    try:
        where_start_pos = lower_query.index(' where ') + len(' where ')
    except ValueError:
        return [] # No WHERE clause found

    # 2. Find the end of the WHERE clause by looking for the next major keyword
    terminators = [' group by ', ' order by ', ' having ', ' limit ', ' union ', ' offset ', ';']
    
    terminator_positions = [lower_query.find(t, where_start_pos) for t in terminators]
    valid_positions = [p for p in terminator_positions if p != -1]
    
    where_end_pos = min(valid_positions) if valid_positions else len(query)

    # 3. Extract the full, correct content of the WHERE clause
    where_clause_content = query[where_start_pos:where_end_pos].strip()

    # 4. Now, safely split the content by 'AND' or 'OR'
    # We will manually split to handle nested parentheses if needed, but for now,
    # a simple split on the full clause content is much safer.
    # We use a regex with word boundaries `\b` just for the split, which is safe now.
    import re
    filters = re.split(r'\s+\b(and|or)\b\s+', where_clause_content, flags=re.IGNORECASE)
    
    # The split includes the delimiters ('and', 'or'), so we take every second element
    conditions = [f.strip() for f in filters[::2] if f.strip()]
    
    return conditions

# Initialize session state variables if not already set
if "table_names" not in st.session_state:
    st.session_state["table_names"] = db.get_table_names()

if "table_info" not in st.session_state:
    st.session_state["table_info"] = db.get_table_info()

# Initialize the database connection and store it in session state
if "sqlalchemy_conn" not in st.session_state:
    _, sqlalchemy_conn = get_sql_database()
    st.session_state["sqlalchemy_conn"] = sqlalchemy_conn
else:
    sqlalchemy_conn = st.session_state["sqlalchemy_conn"]

### Customer Data 

# This function safely runs a query using a temporary, fresh connection.
@st.cache_resource
def run_query(query: str) -> pd.DataFrame:
    """Executes a SQL query and returns a pandas DataFrame."""
    # The 'with' statement ensures the connection is fetched from the pool
    # and returned, even if errors occur.
    with sql_engine.connect() as connection:
        return pd.read_sql(query, connection)

@st.cache_resource
def get_cached_chain():

    llm = get_azure_openai_client()

    sql_prompt_with_memory = ChatPromptTemplate.from_messages([
    # 1. System Message: This contains your instructions, schema, and examples.
    ("system", 
        """
        You are an intelligent SQL assistant. Your task is to generate a syntactically correct [YOUR_SQL_DIALECT] query based on the user's question.

        **DO NOT** answer the question directly.
        **ONLY** output a SQL query. Nothing else. No explanations, no conversation.
        To start you should ALWAYS look at the tables in the database to see what you can query. 
        Do NOT skip this step. Only use the tables and columns explicitly exposed. Do not guess or invent table/column names.

        ### Business Explanation 
        [Describe your database a bit and provide insider-terminology to give the LLM a better understanding of the data.]

        ### Alias Mappings:

        [This section contains the alias mappings for the columns and tables in your database. Use these aliases to refer to the columns and tables in your SQL queries.]
        [Below i theoretical example]
        - "Product Code"
        
        ### Customer Mapping: 
        [Here you can provide customer mappings to map the database customer names to abbrevations]

        ### Special Instructions
        [Here you can add special instructions including when to use which column, or what to do when handling data from different countries with different currencies]
        
        ### Special Query Instruction: 
        # Top X results for multiple groups
        1. For different groups (e.g., countries, years, months), retrieve the top X items (e.g., sold products or customers) ordered by a relevant metric (e.g., sales turnover).
        2. Use the `UNION ALL` operator to combine results from different groups.
        3. Ensure that the results are correctly ordered and limited to the top X items for each group.

        ### Sample Queries
        [Pivotal step: provide example queries to teach the LLM how to write SQL queries]
        [Start from simple queries, but also include difficult queries that require multiple joins, aggregations, and filters.]
        1. Get the aggregated sales and quantity for XXX from 2024 for XXX products
        User Instruction: Get the aggregated sales and quantity for XXX from 2024 for XXX products
        Generated SQL Query:

        SELECT.....
        FROM 
        WHERE ...
        GROUP BY ....


        Be strict. Use only the allowed tables and columns from the list above. If something is unclear, return an error or ask for clarification.
        Only provide the SQL query. Do not include explanations or markdown formatting and remove any "\n" characters from the query.
        Also remove these characters from the quer: "```". Output only the RAW SQL query.
        Only apply filters (e.g., customer name, year, month) if they are *explicitly mentioned* in the user question.
        NEVER round numeric values. Keep them as they are.
        WHENEVER you perform a division, account for division by 0 to prevent division by 0 errors.
        Never use the LIMIT keyword, always use TOP instead. We need to use Microsoft SQL Server syntax.

        ### Follow-Up Questions
        If the user asks a follow-up question, use the context from the chat history to understand their intent.
            
        For example:
        History: [Human: "Get me the top 10 customers in XXX in 2025.", 
        AI: " SELECT TOP(10) ...
                FROM 
                LEFT JOIN 
                ON 
                WHERE 
                GROUP BY 
                ORDER BY 
        New Question: "now for XXX but 2024"
        Your generated SQL should be: 
                SELECT ...             
                WHERE
        
        Always use the table schema below: 
        {schema}
    """),
    
    # 2. Messages Placeholder: This is where the actual conversation history will be injected.
    MessagesPlaceholder(variable_name="chat_history"),

    # 3. Human Message: This is for the user's current question.
    ("human", "{question}"),
    ])

    #prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    def _apply_customer_aliases(sql: str) -> str:
        for pattern, replacement in CUSTOMER_ALIASES:
            sql = re.sub(pattern, replacement, sql)
        return sql
    
    #### This function prevents token limits in the LLM output 
    #### For long SQL responses, it truncates the message to a maximum number of tokens
    def truncate_message(message, max_tokens):
        tokens = message.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return message
    
    CUSTOMER_ALIASES = [
    (r"(?i)'Customer ABC Sapporo", "Customer ABC Sales Sapporo"),    
    (r"(?i)'Customer ABC Fukuoka", "Customer ABC Sales Fukuoka")]

    def process_query(vars_dict: dict) -> dict:
        """
        Cleans and executes a SQL query generated by the LLM.
        It uses the globally available 'sql_engine' to create fresh connections.
        """
        if not isinstance(vars_dict, dict):
            raise TypeError(f"Expected dict in process_query, got: {type(vars_dict)}")

        raw_query = vars_dict.get("query", "")
        st.session_state["raw_query"] = raw_query

        # Your aliasing logic remains the same
        raw_query = _apply_customer_aliases(raw_query) 
        st.session_state["raw_query_customer_alias"] = raw_query

        clean_query = ""

        try:
            ## this whole function serves the purpose of cleaning the SQL query from the LLM output

            query_trimmed = raw_query.strip().lower()
            query_trimmed = query_trimmed.replace("```", "")

            # Check if query starts cleanly and ends with semicolon
            if (query_trimmed.startswith("with") or query_trimmed.startswith("select")) and raw_query.strip().endswith(";"):
                clean_query = raw_query.replace("\n", " ").strip()

            else:
                # Find first occurrence of either 'with' or 'select'
                lower_query = raw_query.lower()
                with_index = lower_query.find("with")
                select_index = lower_query.find("select")

                # Decide which keyword comes first
                if with_index != -1 and (select_index == -1 or with_index < select_index):
                    start_index = with_index
                elif select_index != -1:
                    start_index = select_index
                else:
                    error_message = f"Query does not contain a valid SELECT or WITH clause. Received: '{query_trimmed}'"
                    raise ValueError(error_message)

                semicolon_index = raw_query.find(";") + 1 if ";" in raw_query else len(raw_query)
                clean_query = raw_query[start_index:semicolon_index].replace("\n", " ").strip()

            # Remove trailing semicolon, if present
            if clean_query.endswith(";"):
                clean_query = clean_query[:-1].strip()

            # Execute the query - safer using SQLAlchemy text()

            st.session_state["last_query"] = clean_query
            logging.info(f"Executing cleaned query: {clean_query}")

            response = db.run(clean_query)
            st.session_state["last_response"] = response

            # 2. This is the most critical fix. We use the engine to get a 
            #    temporary connection for the pandas query.
            with sql_engine.connect() as connection:
                df = pd.read_sql(clean_query, connection)
            
            st.session_state["last_dataframe"] = df

            return {**vars_dict, "query": clean_query, "response": response, "dataframe": df}

        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            st.error("⚠️ An error occurred while executing the SQL query.")
            st.code(clean_query, language="sql")
            st.exception(e)

            return {
                **vars_dict,
                "query": clean_query,
                "response": f"Error executing query: {str(e)}",
                "dataframe": pd.DataFrame()
            }

    answer_prompt = PromptTemplate.from_template("""
    Given the following user question, SQL query, and query result, provide a direct answer to the question based on the data.

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}

    Important instructions:
    1. DO NOT describe the SQL query itself
    2. DO NOT start with phrases like "Based on the SQL query..."
    3. DO NOT explain how the data was retrieved
    4. Simply present the requested information in a clear, direct format
    5. If the question asks for a list of items, provide the items as a numbered list
    6. Include relevant numbers and specifics from the data
    7. If the data shows no results, clearly state that no data was found and the user should check for spelling mistakes
    8. If not otherwise specified, return all results. NEVER return more than 15 rows.
    9. If Output is a list, then order it either by alphabet or by a relevant metric (e.g., sales amount, quantity)
    10. To ensure clarity, ALWAYS add a currency symbol to the output. If XXX was used in the SQL query, ALWAYS use EUR or "€", if the query used XXX, use the local currency.
    11. NEVER round numeric values. Keep them as they are. 
    12. For large numeric values, use a comma as a thousands separator (e.g., 1,000,000) and a period as a decimal separator (e.g., 1,234.56)
    Answer:
    """)

    answer_chain = answer_prompt | llm | StrOutputParser()

    sql_chain = (
        sql_prompt_with_memory
        | llm.bind(stop="\nSQL Result:")
        | StrOutputParser()
    )

    # --- FULL CHAIN DEFINITION ---
    # This chain is now fully self-contained.
    full_chain = (
        RunnablePassthrough.assign(
            # This lambda now gets the memory object from the input dictionary 'x'
            chat_history=RunnableLambda(lambda x: x["memory"].load_memory_variables(x)["chat_history"])
        )
        | RunnablePassthrough.assign(
            query=(
                RunnablePassthrough.assign(
                    schema=lambda x: db.get_table_info()
                )
                | sql_chain
            )
        )
        | RunnableLambda(process_query) # Assuming process_query is defined
        | RunnableLambda(lambda vars: {
            **vars,
            "answer": answer_chain.invoke({
                "question": truncate_message(vars["question"], 1000),
                "query": truncate_message(vars["query"], 1000),
                "response": truncate_message(vars["response"], 1000),
                "chat_history": vars["chat_history"]
            })
        })
    )
    
    return full_chain

full_chain = get_cached_chain()

######### Display App 

### Title 

st.title("🧠 SQL Assistant")

### Warning Message before using 

if "warning_dismissed" not in st.session_state:
    st.session_state.warning_dismissed = False

if not st.session_state.warning_dismissed:
    with st.container(border=True):
        st.warning("⚠️ Important: Please Read Before Using")

        st.markdown("""
        - **Generative AI Disclaimer:** You are using a generative AI tool. **All outputs should be carefully checked for accuracy.**
        - **Data Timeliness:** The database contains historical data and may not be a reliable source for real-time, live information.
        - **Data Completeness:** Not all data from XXX & ZZZ is reflected here.
        - **Data Security:** The data in this application is classified as XXX. Do not share sensitive or confidential information with others who have no access rights.
        - **Feedback & Support:** For feedback or specific data requests, please reach out to [gablzub@example.com](mailto:gablzub@example.com)
        """)

        if st.button("I understand, close this message", key="dismiss_warning"):
            st.session_state.warning_dismissed = True
            st.rerun()

# The rest of the app loads only after the warning is dismissed
if st.session_state.warning_dismissed:
        # Streamlit UI starts here
    # ---------------------------
    with st.popover("💡 Show Example Prompts"):
            st.subheader("Example Prompts")
            st.write("You can use the following example prompts to get started. Be aware that you need to be **specific**. Don't forget to mention a **XXX**, **XXX** and the **time period** when necessary.")
            st.write("Customer related prompts:")
            
            st.info("Can you get me the top 10 customers in XXX with the highest decrease in sales revenue in 2025 compared to the previous year?")

            st.info("")

            st.write("Product related prompts:")

            st.info("For all active XXX products with XXX status, get me the sales in 2025. Add ... to the data.")

            st.write("Click anywhere outside this box to close it.")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)

    user_input = st.chat_input("Ask a question about XXX:")

    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.spinner("Generating and running query... 🍵"):
            success = False
            response = None
            for attempt in range(3):
                try:
                    chain_input_with_memory = {
                    "question": user_input,
                    "memory": st.session_state.memory  # Get the memory object from session state
                        }
                    response = full_chain.invoke(chain_input_with_memory)
                    success = True
                    break
                except APIConnectionError:
                    st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                    time.sleep(2)

        if success:

            ## Add memory 

            st.session_state.memory.save_context(
                {"input": user_input},
                {"output": response["answer"]}
            )

            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            st.session_state["last_query"] = response.get("query", "")
            st.session_state["last_response"] = response

            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(response["answer"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.error("❌ Failed to connect after 3 attempts. Please check your internet or try again later.")

################################################################
# Sidebar (outside tabs, stays visible)

@st.cache_resource
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

with st.sidebar:
    
    st.write("### 📌 Current Query")
    # Access the variables stored in Streamlit's session state

    if "last_query" in st.session_state and st.session_state["last_query"]:
        st.markdown("#### Last SQL Query:")

        with st.expander("Last SQL Query (click to expand)"):
            st.write(st.session_state["last_query"])

        last_response = st.session_state.get("last_response", {})
        df = last_response.get("dataframe") if last_response else None

        if df is not None and not df.empty:
            excel_bytes = to_excel_bytes(df)
            st.download_button(
                label="⬇️ Download Last Query Result as Excel",
                data=excel_bytes,
                file_name="last_query_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.write("No queries yet.")

    # Display list of tables
    st.write("### 🗃️ Available Tables")

    if "table_names" in st.session_state and st.session_state.table_names:
        with st.expander("Click to view tables"):
            for table in st.session_state.table_names:
                st.markdown(f"- `{table}`")
    else:
        st.write("No tables loaded.")

    if "last_query" in st.session_state and st.session_state["last_query"]:
        st.write("### 📝 Last Query Filters")

        query_filters = get_sql_filters(st.session_state["last_query"])
        with st.expander("Click to view query filters"):
            for filter in query_filters:
                st.markdown(f"- `{filter}`")

    else:
        st.write("No queries yet.")

    # Sidebar - Reset session button
    if st.button("🔄 Reset Chat & Clear Cache"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_resource.clear()  
        st.rerun()
