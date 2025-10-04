
### standard
import os
import io
import streamlit as st
import pandas as pd
import re 
import warnings
from openai import APIConnectionError
import time

### .env & langchain
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import AzureChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda

from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, START, END
from langchain.schema import BaseMessage

import urllib
from sqlalchemy import create_engine, inspect, text

### Sharepoint API

import streamlit as st
import pandas as pd
import requests
import tempfile
from datetime import datetime
import threading

### Authentication 

import streamlit_authenticator as stauth
import yaml # New import
from yaml.loader import SafeLoader # New import

########################################### Chatbot #####################################################

# Must be the first Streamlit command in the script.
st.set_page_config(
    page_title="SQL Assistant",  # The title in the browser tab
    page_icon="🧠",             # The icon in the browser tab
    layout="wide"                # Optional: Use wide layout
)


LOGO_RELATIVE_PATH = "images/streamlit-mark-dark.png"
st.image(LOGO_RELATIVE_PATH, width=180)

######################## Agent Streamlit Interface ########################

warnings.filterwarnings("ignore")


# Initialize session state variables BEFORE page UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Welcome to the SQL Assistant!") ## [Adjust Initial Text]
    ]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "page" not in st.session_state:
    st.session_state.page = "💬 SQL Assistant Chatbot" # Default page 

# For storing the last generated SQL query
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# For storing the last response object from the LLM, which includes the result and dataframe
if "last_dataframe" not in st.session_state:
    st.session_state["last_dataframe"] = pd.DataFrame()

# For storing chat messages (Even though not explicitly in the sidebar, a chatbot needs this)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_used_node" not in st.session_state:
    st.session_state["last_used_node"] = ""

### Load Environment
load_dotenv()

### .db credentials for sharepoint [Adjust if you don't have a .db object (SQLite) on the Sharepoint]
### Note: any excel file (maintained regularly) can be transformed into a .db file using python locally and then uploaded to Sharepoint

# Ensure these environment variables are set in your Cloud Foundry deployment
client_secret = os.environ.get('SHAREPOINT_CLIENT_SECRET')
tenant_id = os.environ.get('SHAREPOINT_TENANT_ID')
client_id = os.environ.get('SHAREPOINT_CLIENT_ID')
resource = 'https://graph.microsoft.com/'F
grant_type = 'client_credentials'

# The site ID and file information
site_id = 'XXX.sharepoint.com' ## [Adjust Sharepoint site ID]
folder_path = 'xxxx/chatbot_folder' ## [Adjust Sharepoint folder path]
db_file_name = 'xxx.db' ## [Adjust to your actual .db name]

# Cache the Azure OpenAI client
@st.cache_resource
def get_azure_openai_client():
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")  # Required for LangChain
    os.environ["OPENAI_API_TYPE"] = "azure"
    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        request_timeout=840  # ⬅️ Timeout set to 5 minutes
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

@st.cache_resource ##### Function to enable column filtering: manipulates the get_info output
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

## SQL data

@st.cache_resource
def get_colnames(sql_statements):
    # Split the SQL statements into individual table definitions
    table_definitions = sql_statements.strip().split('CREATE TABLE ')[1:]

    data = []
    for table_def in table_definitions:
        lines = table_def.split('\n')
        table_name = lines[0].strip().split(' ')[0]
        for line in lines[1:]:
            if line.strip() and not line.strip().startswith(')'):
                column_name = line.strip().split(' ')[0]
                data.append([table_name, column_name])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Table Name', 'Column Name'])

        # Add Source column based on table names
    source_mapping = { 
        'xxx': 'Source1', ## [Adjust to your actual table names on the left, on the right the Source System such as SAP or local excel etc]
        'xxx': 'Source2',
    }

    df['Source'] = df['Table Name'].map(source_mapping).fillna('Unknown')
    
    return df

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

    # Use table list 
    table1 = ["tab1"]
    table2 = ["tab2"] ## [Adjust your table names]

    ######## Create databases 
    ## tab1

    tab1_cols = ["Col1", "Col2"] ## [Adjust your column names]


    db_tab1 = SQLDatabase( ## [Adjust db name]
        engine=engine,
        include_tables=table1, ## [Adjust table name]
        schema="schema1", ## [Adjust your schema name]
        sample_rows_in_table_info=2
    )

    filtered_table1 = filter_columns_multi(
        schema_text=db_tab1.get_table_info(),
        tables=table1,
        allowed_columns=tab1_cols
    )

    # This is just to make the metadata appear more user-friendly
    db_tab1.get_table_info = lambda *_, **__: filtered_table1

    ## tab2

    tab2_cols = ["Col3", "Col4"] ## [Adjust your column names]


    db_tab2 = SQLDatabase(
        engine=engine,
        include_tables=table2,
        schema="schema2", ## [Adjust your schema name]
        sample_rows_in_table_info=2
    )

    filtered_table2 = filter_columns_multi(
        schema_text=db_tab2.get_table_info(),
        tables=table2,
        allowed_columns=tab2_cols
    )

    db_tab2.get_table_info = lambda *_, **__: filtered_table2


    ############# Now combine into one dummy db 

    combined_table_info  = "\n\n".join(
        [db_tab1.get_table_info(), db_tab2.get_table_info()]
    )

    combined_table_names = db_tab1.get_table_names() + db_tab2.get_table_names() 

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

### same for .db object: you might need to adjust the code here
# ## First prepare function for adjusting db info 
@st.cache_resource
def clean_sql_string(sql_str):
    # Replace literal escape characters with their actual representations
    sql_str = sql_str.replace('\\n', '\n').replace('\\t', '\t')
    
    # Split into lines for processing
    lines = sql_str.split('\n')
    
    # Process CREATE TABLE statement
    create_table_section = []
    comment_section = []
    in_comment = False
    
    for line in lines:
        if line.strip().startswith('/*'):
            in_comment = True
            comment_section.append(line)
        elif in_comment:
            comment_section.append(line)
            if line.strip().endswith('*/'):
                in_comment = False
        else:
            create_table_section.append(line)
    
    # Format CREATE TABLE statement
    formatted_create = '\n'.join(create_table_section)
    
    # Format comment section to maintain readability
    formatted_comment = '\n'.join(comment_section)
    
    return formatted_create + '\n\n' + formatted_comment 

######################################################################### Change from Local parquet (or excel or any tabular data) to direct .db access on sharepoint ################################################################33

## Load .db
# --- SharePoint Database Download Function ---
@st.cache_resource
def download_db_from_sharepoint():
    try:
        # Get access token
        token_api = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
        payload = f'grant_type={grant_type}&client_id={client_id}&client_secret={client_secret}&resource={resource}'
        response_token = requests.request("POST", token_api, data=payload, verify=True, timeout=840) # Added timeout
        token = response_token.json()['access_token']

        # Get the database file metadata
        file_api = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{folder_path}/{db_file_name}"
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.request("GET", file_api, headers=headers, verify=True, timeout=840) # Added timeout
        file_data = response.json()
        download_url = file_data.get('@microsoft.graph.downloadUrl')

        if not download_url:
            st.error(f"Could not get download URL for {db_file_name}.")
            st.json(file_data) # Show the response for debugging
            return None

        # Download the database file content
        db_content_response = requests.get(download_url, timeout=840) # Increased timeout for large file
        db_content = db_content_response.content

        # Generate a unique temporary path for the downloaded DB
        # Using a more robust temp file creation for safety
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
            temp_file.write(db_content)
            temp_db_path = temp_file.name

        return temp_db_path

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please check your internet connection or increase timeout values.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing SharePoint: {e}")
        st.info("Please check your SharePoint credentials and network connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during SharePoint download: {e}")
        return None

## Write csv file to track queries 
## Note: this assumes the CSV is already existing in the specified folder (same as .db object)
@st.cache_resource
def save_chat_data_to_sharepoint(question, query, used_node, answer=None):
    """
    Save chat data to chatbot_tracker.csv on SharePoint by downloading, appending, and re-uploading
    """
    try:
        # Configuration for the CSV file
        csv_file_name = "chatbot_tracker.csv" ### [Adjust if your csv is different]
        
        # Get access token
        token_api = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
        payload = f'grant_type={grant_type}&client_id={client_id}&client_secret={client_secret}&resource={resource}'
        response_token = requests.request("POST", token_api, data=payload, verify=True, timeout=840)
        token = response_token.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        # Try to download existing CSV file
        file_api = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{folder_path}/{csv_file_name}"
        response = requests.request("GET", file_api, headers=headers, verify=True, timeout=840)
        
        existing_df = pd.DataFrame()
        
        if response.status_code == 200:
            # File exists, download it
            file_data = response.json()
            download_url = file_data.get('@microsoft.graph.downloadUrl')
            
            if download_url:
                csv_content_response = requests.get(download_url, timeout=840)
                csv_content = csv_content_response.content
                
                # Read existing CSV into DataFrame
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                    temp_file.write(csv_content)
                    temp_csv_path = temp_file.name
                
                try:
                    existing_df = pd.read_csv(temp_csv_path)
                except pd.errors.EmptyDataError:
                    existing_df = pd.DataFrame(columns=['Date', 'Question', 'Query', 'Node', 'Answer'])
                
                # Clean up temp file
                os.unlink(temp_csv_path)
        else:
            # File doesn't exist, create DataFrame with correct columns
            existing_df = pd.DataFrame(columns=['Date', 'Question', 'Query', 'Node', 'Answer'])
        
        # Create new row data matching your column structure
        new_row = {
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Question': question,
            'Query': query if query else "",
            'Node': used_node if used_node else "",
            'Answer': answer if answer else ""
        }
        
        # Append new row to existing DataFrame
        new_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save updated DataFrame to temporary CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='', encoding='utf-8') as temp_file:
            new_df.to_csv(temp_file.name, index=False)
            temp_csv_path = temp_file.name
        
        # Upload updated CSV back to SharePoint
        upload_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{folder_path}/{csv_file_name}:/content"
        
        with open(temp_csv_path, 'rb') as file_content:
            upload_response = requests.put(
                upload_url,
                headers=headers,
                data=file_content,
                timeout=840
            )
        
        # Clean up temp file
        os.unlink(temp_csv_path)
        
        if upload_response.status_code in [200, 201]:
            return True
        else:
            st.warning(f"Failed to upload CSV to SharePoint. Status: {upload_response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Error saving chat data to SharePoint: {e}")
        return False


# --- Main Chatbot Data Loading Functions ---

# This function will now download the DB and provide its path
@st.cache_resource
def get_sqlite_db_path():
    db_file_path = download_db_from_sharepoint()
    if db_file_path is None:
        st.stop() # Stop execution if download fails
    return db_file_path

# Get the path to the downloaded database (this will trigger the download)
sqlite_db_path = get_sqlite_db_path()

# This function now takes the path to the downloaded SQLite DB
@st.cache_resource
def get_data_from_sqlite(db_path: str):

    try:
        # Create a SQLite database connection using the downloaded file
        # Use an absolute path for reliability
        engine_sqlite = create_engine(f"sqlite:///{db_path}", echo=False)

        ################################# code below should have no change ###########################################################

        db_tables = ["your_table"] ## [Adjust your table names]

        # Create the SQLDatabase object
        db_sqlite= SQLDatabase(
            engine=engine_sqlite,
            include_tables=db_tables,
            sample_rows_in_table_info=2
        )

        db_sqlite_info = db_sqlite.get_table_info()
        db_sqlite_tables = db_sqlite.get_table_names()

        db_sqlite_info = clean_sql_string(db_sqlite_info)

        # Note: These lambda assignments are a bit hacky. LangChain might prefer direct access.
        db_sqlite.get_table_names = lambda *_, **__: db_sqlite_tables
        db_sqlite.get_table_info  = lambda *_, **__: db_sqlite_info

        return db_sqlite, engine_sqlite
    except Exception as e:
        st.error(f"Error setting up SQLite database for LangChain: {e}")
        st.stop() # Stop the app if DB setup fails


# Call get_data_from_sqlite, passing the path to the downloaded DB
# This is the corrected call
db_sqlite, engine_sqlite = get_data_from_sqlite(sqlite_db_path)

######################################################################### Change from Local parquet to direct .db access on sharepoint ################################################################33

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

#### Database summary 

# 1. Store SQL table info in session state (if not already there)
if "table_info_sql" not in st.session_state:
    st.session_state["table_info_sql"] = get_colnames(db.get_table_info())

    # 2. Function to generate sqlite data info.
@st.cache_resource
def get_sqlite_data_info(_sql_engine, table_name="your_table"): ## [Adjust your table name]

    try:
        # Use sqlalchemy.inspect for more reliable schema parsing
        inspector = inspect(_sql_engine) 
        
        columns = inspector.get_columns(table_name)

        column_names = [col['name'] for col in columns]

        df_table_info = pd.DataFrame({
            "Column Name": column_names
        })

        df_table_info['Source'] = "your_source" ## [Adjust to your actual source system]
        df_table_info['Table Name'] = table_name
        return df_table_info

    except Exception as e:
        # Include the table_name in the error message for better debugging
        st.error(f"Error getting column info for table '{table_name}': {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# 3. Get the .db data info (this will hit the cache after the first run)
df_sqlite_info = get_sqlite_data_info(engine_sqlite, table_name="your_table") ## [Adjust your table name]

# 4. Combine with SQL table info and store in session state (if not already there)
if "combined_table_info" not in st.session_state:
    st.session_state["combined_table_info"] = pd.concat(
        [st.session_state["table_info_sql"], df_sqlite_info],
        ignore_index=True
    )

### Customer Data 

# This function safely runs a query using a temporary, fresh connection.
@st.cache_resource
def run_query(query: str) -> pd.DataFrame:
    """Executes a SQL query and returns a pandas DataFrame."""
    # The 'with' statement ensures the connection is fetched from the pool
    # and returned, even if errors occur.
    with sql_engine.connect() as connection:
        return pd.read_sql(query, connection)

########################################### Graph Chains #####################################

#### Standard SQL Chain

@st.cache_resource
def get_cached_chain():

    llm = get_azure_openai_client()

    sql_prompt_with_memory = ChatPromptTemplate.from_messages([
        # 1. System Message: This contains your instructions, schema, and examples.
        ("system", 
        """
        <Instructions>
        You are an intelligent SQL assistant.

        **DO NOT** answer the question directly.
        **ONLY** output a SQL query. Nothing else. No explanations, no conversation.
        To start you should ALWAYS look at the tables in the database to see what you can query. 
        Do NOT skip this step. Only use the tables and columns explicitly exposed. Do not guess or invent table/column names.

        [Adjust and add more direct instructions]


        </Instructions>

        <Business Explanation> 
        </Business Explanation>

        <Alias Mappings>
        [Adjust and add your alias mappings here, such as vocabulary that is used for specific columns]
        </Alias Mappings>

        <Customer Mapping>
        [You can provide other variables for prompt injecting here such as the whole customer list to make it easier for the LLM to infer which customer is meant.
        For this, just use curly brackets: {"customer_list"} (remove the quotation marks)
        </Customer Mapping>

        <Special Instructions>
        - Ensure that all generated SQL queries are compatible with the [Your Dialect] dialect.
        </>Special Instructions>

        <Table Explanation>
        </Table Explanation>

        <Table Join Instructions>
        <Table Join Instructions>
        
        <Country data>
        [Just another example of what you might want to specify. E.g., if country A is used, filter a column by ABC]
        </Country data>

        <Special Query Instruction> 
        </Special Query Instruction>


        <Sample Queries>

        1. 
        User Instruction: 
        Generated SQL Query:

        SELECT 
        FROM 
        WHERE 
        GROUP BY 

        [Note: it makes sense to add answer queries to complex queries, e.g. queries that require CTEs or subqueries]
        </Sample Queries>

        <Final Instructions>
        Be strict. Use only the allowed tables and columns from the list above. 
        Only provide the SQL query. Do not include explanations or markdown formatting and remove any "\n" characters from the query.
        Also remove these characters from the quer: "```". Output only the RAW SQL query.
        WHENEVER you perform a division, account for division by 0 to prevent division by 0 errors.
        Never use the LIMIT keyword, always use TOP instead. We need to use Microsoft SQL Server syntax. [Adjust for other dialects, e.g. Oracle SQL]
        </Final Instructions>

        <Follow-Up Questions>
        If the user asks a follow-up question, use the context from the chat history to understand their intent.
            
        For example:
        History: [Human: "xxx", 
        AI: " SELECT ...."]
        New Question: "now for xxx"
        Your generated SQL should be: 
                "SELECT ..."
        </Follow-Up Questions>

        Always use the table schema below: 
        {schema}
    """),
        
        # 2. Messages Placeholder: This is where the actual conversation history will be injected.
        MessagesPlaceholder(variable_name="chat_history"),

        # 3. Human Message: This is for the user's current question.
        ("human", "{question}"),
    ])

    def get_schema(_):
        return db.get_table_info()

    def _apply_customer_aliases(sql: str) -> str:
        for pattern, replacement in CUSTOMER_ALIASES:
            sql = re.sub(pattern, replacement, sql)
        return sql
    
    def truncate_message(message, max_tokens):
        tokens = message.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return message

    CUSTOMER_ALIASES = [] ## [Adjust your customer aliases here as regex patterns]

    def process_query(vars):
        if not isinstance(vars, dict):
            raise TypeError(f"Expected dict in process_query, got: {type(vars)}")

        raw_query = vars.get("query", "")

        raw_query = _apply_customer_aliases(raw_query)

        clean_query = ""

    
        # Log the raw query for debugging
        query_trimmed = raw_query.strip()
        query_trimmed = query_trimmed.replace("```", "")

        # Use lowercase version only for checking keywords
        query_lower = query_trimmed.lower()

        if not query_trimmed or (not "select" in query_lower and not "with" in query_lower):
            return {**vars, 
                    "query": query_trimmed,
                    "response": f"No valid SQL query was provided. Input received: '{query_trimmed}'", 
                    "dataframe": pd.DataFrame()}

        # Check if query starts cleanly and ends with semicolon - use lowercase for startswith check
        if (query_lower.startswith("with") or query_lower.startswith("select")) and query_trimmed.endswith(";"):
            clean_query = query_trimmed.replace("\n", " ").strip()
        else:
            # Find first occurrence of either 'with' or 'select'
            with_index = query_lower.find("with")
            select_index = query_lower.find("select")

            # Decide which keyword comes first
            if with_index != -1 and (select_index == -1 or with_index < select_index):
                start_index = with_index
            elif select_index != -1:
                start_index = select_index
            else:
                return {**vars, 
                    "query": query_trimmed,
                    "response": f"No valid SQL query was provided. Input received: '{query_trimmed}'", 
                    "dataframe": pd.DataFrame()}

            semicolon_index = query_trimmed.find(";") + 1 if ";" in query_trimmed else len(query_trimmed)
            clean_query = query_trimmed[start_index:semicolon_index].replace("\n", " ").strip()

        # Remove trailing semicolon, if present
        if clean_query.endswith(";"):
            clean_query = clean_query[:-1].strip()

        # Execute the query - safer using SQLAlchemy text()

        try:
            with sql_engine.connect() as connection:
                df = pd.read_sql(clean_query, connection)
            response = df.to_markdown(index=False)
            
        except Exception as e:
            error_message = str(e)
            response = f"Error executing SQL query: {error_message}\n\nQuery attempted: {clean_query}"
            df = pd.DataFrame() 

        return {**vars, "query": clean_query, "response": response, "dataframe": df}

    # Answer generation prompt
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
        10. NEVER round numeric values. Keep them as they are. 
        11. NEVER LIE and NEVER make up data, when the SQL query result is empty, tell the user HONESTLY no data was found or to check for spelling mistakes.
        Answer:
        """)

    sql_chain_memory = (
        sql_prompt_with_memory
        | llm.bind(stop="\nSQL Result:")
        | StrOutputParser()
    )

    # Answer generation chain
    answer_chain = answer_prompt | llm | StrOutputParser()

    full_chain_graph = (
        RunnableLambda(lambda x: { # Extract question and chat_history from the state object 'x'
            "question": x["question"],
            "chat_history": x["chat_history"],
            "memory": x["memory"] # Pass the memory object if any sub-component needs it
        })
        | RunnablePassthrough.assign(
            query=(
                RunnablePassthrough.assign(
                    schema=lambda x: db.get_table_info()
                    ##customer_list=lambda x: customer_list [Adjust if you want to insert variables into the prompt]
                )
                | sql_chain_memory # sql_chain_memory expects question, chat_history, schema, customer_lists
            )
        )
        | RunnableLambda(process_query)
        | RunnableLambda(lambda vars: {
            **vars,
            "answer": answer_chain.invoke({
                "question": truncate_message(vars.get("question", ""), 1000),
                "query": truncate_message(vars.get("query", ""), 1000),
                "response": truncate_message(vars.get("response", ""), 1000),
                "chat_history": vars.get("chat_history", []) # Get from `vars`
            })
        })
    )
    
    return full_chain_graph

full_chain = get_cached_chain()

### Explain data chain
## Note: this is just a prompt template that tells the LLM about the data and enables it to answer questions based on the template i

@st.cache_resource
def get_explain_chain():

    llm = get_azure_openai_client()
    table_info = st.session_state["combined_table_info"]

    explanation_template = """
    <Data Summary>
    </Data Summary>


    <Business Explanation> 
    </Business Explanation> 

    <Table Overview>
    Here is the tables overview with table name, column names and sources.
    ALWAYS USE this when users ask for available data in the different sources:
    {table_info}
    </Table Overview>

    <Example Prompts>
    [Adjust to add some examples on how to use the chatbot]
    </Example Prompts>

    <Chat History>
    This is the chat history:
    {chat_history}
    </Chat History>

    <User Question>
    Be brief and concise. No need to explain the whole background if asked an overall question. 
    Answer the question:
    {question}
    </User Question>
    """

    explanation_prompt = ChatPromptTemplate.from_template(explanation_template)

    data_explanation_chain = (
        RunnableLambda(lambda x: { # Extract question and chat_history from the state object 'x'
            "question": x["question"],
            "table_info": table_info, # Ensure table_info is accessible
            "chat_history": x["chat_history"]
        })
        | explanation_prompt
        | llm
        | StrOutputParser()
    )

    return data_explanation_chain

explain_chain = get_explain_chain()


#### SQLite Chain for the graph

@st.cache_resource
def get_sqlite_chain(): 
    llm = get_azure_openai_client()

    sqlite_prompt_with_memory = ChatPromptTemplate.from_messages([
        # 1. System Message: This contains your instructions, schema, and examples.
        ("system", 
        """
        <Instructions>
        You are an intelligent SQL assistant.

        When users ask questions using business terms, translate them into actual SQL queries by mapping their terms to database columns and tables as shown below.
        To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step. Only use the tables and columns explicitly exposed. Do not guess or invent table/column names.
        ALWAYS use the SQLite SQL dialect.
        </Instructions>

        <Business Explanation> 
        </Business Explanation>
        
        <Alias Mappings>
        </Alias Mappings>
        
        <Special Instructions>
        </Special Instructions>

        <Example Queries>
            1. User Input: ""
            Generated Query:
                SELECT
                    ...
        </Example Queries>

        <Additional Instructions>
        Be strict. Use only the allowed tables and columns from the list above. If something is unclear, return an error or ask for clarification.
        Only provide the SQL query. Do not include explanations or markdown formatting and remove any "\n" characters from the query.
        Also remove these characters from the query: "```". Output only the RAW SQL query.
        Only apply filters if they are *explicitly mentioned* in the user question.
        NEVER round numeric values. Keep them as they are.
        WHENEVER you perform a division, account for division by 0 to prevent division by 0 errors.
        Never use the TOP keyword, always use LIMIT instead. We need to use SQLite syntax.
        </Additional Instructions>

        <Follow-Up Questions>
        If the user asks a follow-up question, use the context from the chat history to understand their intent.
            
        For example:
        History: [Human: "xxx", 
        AI: "
            SELECT...
        "]

        New Question: "Now for xxx"
        Your generated SQL should be: 
            SELECT ...
        </Follow-Up Questions>

        Always use the table schema below: 
        {schema}
    """),
        
        # 2. Messages Placeholder: This is where the actual conversation history will be injected.
        MessagesPlaceholder(variable_name="chat_history"),

        # 3. Human Message: This is for the user's current question.
        ("human", "{question}"),
    ])

    sqlite_chain_memory = (
    RunnableLambda(lambda x: {
            "question": x["question"],       # Current question
            "chat_history": x["chat_history"], # History from previous steps in the chain
            "schema": x["schema"]            # Schema from the RunnablePassthrough.assign
        })
        | sqlite_prompt_with_memory
        | llm.bind(stop="\nSQL Result:")
        | StrOutputParser()
    )

        # Answer generation prompt
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
        10. NEVER round numeric values. Keep them as they are. 
        11. When presenting numerical values, format them with a comma as the thousand separator (e.g., 1,800,000).
        12. NEVER LIE and NEVER make up data, when the SQL query is empty, just tell the user no data was found or to check for spelling mistakes.
        Answer:
        """)
    
    answer_chain = answer_prompt | llm | StrOutputParser()

    def truncate_message_sqlite(message, max_tokens):
        # Handle DataFrame
        if isinstance(message, pd.DataFrame):
            # Convert DataFrame to string representation
            message_str = message.to_string()
        # Handle None
        elif message is None:
            return ""
        # Handle other non-string types
        elif not isinstance(message, str):
            message_str = str(message)
        else:
            message_str = message
            
        tokens = message_str.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return message_str

    def process_query_sqlite(vars):
        if not isinstance(vars, dict):
            raise TypeError(f"Expected dict in process_query, got: {type(vars)}")

        raw_query = vars.get("query", "")

        clean_query = ""

    
        # Log the raw query for debugging
        query_trimmed = raw_query.strip()
        query_trimmed = query_trimmed.replace("```", "")

        # Create lowercase version for keyword checking
        query_lower = query_trimmed.lower()

        # Handle empty or non-query text
        if not query_trimmed or (not "select" in query_lower and not "with" in query_lower):
            return {**vars, 
                    "query": query_trimmed, 
                    "response": f"No valid SQL query was provided. Input received: '{query_trimmed}'", 
                    "dataframe": pd.DataFrame()}

        # Check if query starts cleanly and ends with semicolon - use lowercase for startswith
        if (query_lower.startswith("with") or query_lower.startswith("select")) and query_trimmed.endswith(";"):
            clean_query = query_trimmed.replace("\n", " ").strip()
        else:
            # Find first occurrence of either 'with' or 'select' in lowercase
            with_index = query_lower.find("with")
            select_index = query_lower.find("select")

            # Decide which keyword comes first
            if with_index != -1 and (select_index == -1 or with_index < select_index):
                start_index = with_index
            elif select_index != -1:
                start_index = select_index
            else:
                return {**vars, 
                        "query": query_trimmed, 
                        "response": f"No valid SQL query was provided. Input received: '{query_trimmed}'", 
                        "dataframe": pd.DataFrame()}

            semicolon_index = query_trimmed.find(";") + 1 if ";" in query_trimmed else len(query_trimmed)
            clean_query = query_trimmed[start_index:semicolon_index].replace("\n", " ").strip()

        # Remove trailing semicolon, if present
        if clean_query.endswith(";"):
            clean_query = clean_query[:-1].strip()

        # Execute the query - safer using SQLAlchemy text()
        try:
            with engine_sqlite.connect() as connection:
                df = pd.read_sql(clean_query, connection)
            response = df.to_markdown(index=False)
        except Exception as e:
            error_message = str(e)
            response = f"Error executing SQL query: {error_message}\n\nQuery attempted: {clean_query}"
            df = pd.DataFrame() 
            
        response = df.to_markdown(index=False)

        return {**vars, "query": clean_query, "response": response, "dataframe": df}
    
        # chain_graph
    sqlite_chain_graph = (
        RunnableLambda(lambda x: { # Extract question and chat_history from the state object 'x'
            "question": x["question"],
            "chat_history": x["chat_history"],
            "memory": x["memory"] # Pass the memory object if any sub-component needs it (e.g. for loading variables)
        })
        | RunnablePassthrough.assign( # This PassThrough is to ensure the query sub-chain gets question and chat_history
            query=(
                RunnablePassthrough.assign(
                    schema=lambda x: db_sqlite.get_table_info()
                )
                | sqlite_chain_memory 
            )
        )
        | RunnableLambda(process_query_sqlite) # This lambda will receive {question, chat_history, query, response}
        | RunnableLambda(lambda vars: {
            **vars,
            "answer": answer_chain.invoke({
                "question": truncate_message_sqlite(vars["question"], 1000),
                "query": truncate_message_sqlite(vars["query"], 1000),
                "response": truncate_message_sqlite(vars["response"], 1000),
                "chat_history": vars["chat_history"]  # Pass the history to the answer chain from `vars`
            })
        })
    )
    
    return sqlite_chain_graph

sqlite_chain = get_sqlite_chain()

## Now the router node agent

@st.cache_resource
def get_router_agent():
    llm = get_azure_openai_client()

        # Define the router prompt
    router_prompt = ChatPromptTemplate.from_template("""
    <Instructions>
     You are a intelligent router that decides which tool to use based on the user question and chat history.
    </Instructions>

    <Tools>                                             
        The options are:
        - "sql": This ACTUALLY gets the data from the SQL database. Used for questions requiring SQL queries, data analysis, or specific information from the database. In this database you can find data from ... [Adjust to your actual database and data sources]
        - "sqlite": This ACTUALLY gets the data from the XXX database. Used for questions related to the xxx [Adjust to your actual database and data sources]
        - "explain": Just tells the user a bit about the data & sources, it can NOT get data from the databases. Used for simple questions about data schema, explanations, or what data is available. No SQL queries are needed here. 
    </Tools>

    <Tool Usage>
        Examples when to use which tool:
        sql: 
        -                                             
                                                    
        explain:
        - "What data is available in the database?"
        - "Tell me about the available data and its source"
        - "Get me all available columns with source please"
        - "Where is the data coming from"       

                                            

        sqlite: 
        - 
    </Tool Usage> 
                                                     
    <Special Instruction>
    </Special Instruction>

    <Follow-Up Questions>
    If the user asks a follow-up question, use the context from the chat history to understand their intent.
        
    For example:
    History: [Human: "", 
    AI: "
        SELECT ...",
    "]

    New Question: "Now check ..."
    Your generated SQL should be: 
        SELECT ...;
    </Follow-Up Questions>   
                                         
    <User question>
    {question}
    </User question>
    
    <Chat history>
    {chat_history}
    </Chat history>
    Based on the question and chat history, choose the best option: sql, explain, or sqlite.
    Respond ONLY with one of these options exactly: sql or explain or sqlite.
    """)

    router_chain = (
        RunnableLambda(lambda x: { # Extract question and chat_history from the state object 'x'
            "question": x["question"],
            "chat_history": x["chat_history"]
        })
        | router_prompt 
        | llm
        | RunnableLambda(lambda x: x.content.strip().lower())
    )

    return router_chain

router_chain = get_router_agent()

############################################################# Graph Setup ####################################################################

# --- Define the graph state ---
class ChatState(TypedDict):
    question: str
    memory: ConversationBufferMemory   
    chat_history: list[BaseMessage]
    answer: Optional[str]
    query: Optional[str]
    dataframe: Optional[Any]
    used_node: Optional[Any]


# --- Tool wrappers ---
def run_full_chain(state: ChatState) -> ChatState:
    full_result = full_chain.invoke(state) # Pass the whole 'state' object

    answer = full_result["answer"]

    state["answer"] = answer
    state["query"] = full_result.get("query")
    state["dataframe"] = full_result.get("dataframe")
    
    # Append the AI's response to the graph state's chat_history for display/future nodes
    state["chat_history"].append(AIMessage(content=answer))
    
    # Update the PERSISTENT memory object with the completed turn
    state["memory"].save_context({"input": state["question"]}, {"output": answer})

    state["used_node"] = "sql"
    
    return state


def run_data_explanation_chain(state: ChatState) -> ChatState:
    # data_explanation_chain returns a string directly
    raw_answer_string = explain_chain.invoke({
        "question": state["question"],
        "chat_history": state["chat_history"] # Pass the correctly prepped history from ChatState
    })

    # The 'answer' for the state will be this raw string
    state["answer"] = raw_answer_string
    
    # Append the AI's response to the graph state's chat_history for display/future nodes
    state["chat_history"].append(AIMessage(content=raw_answer_string))

    # Update the PERSISTENT memory object with the completed turn
    state["memory"].save_context(
        {"input": state["question"]},
        {"output": raw_answer_string} # Save the raw string as the output
    )

    state["used_node"] = "explain"

    return state

def run_sqlite_chain(state: ChatState) -> ChatState:

    # REVERT: Pass the entire 'state' object to the chain
    full_result = sqlite_chain.invoke(state) # Pass the whole 'state' object

    answer = full_result["answer"]
    
    state["answer"] = answer
    state["query"] = full_result.get("query")
    state["dataframe"] = full_result.get("dataframe")
    
    # Append the AI's response to the graph state's chat_history for display/future nodes
    state["chat_history"].append(AIMessage(content=answer))
    
    # Update the PERSISTENT memory object with the completed turn
    state["memory"].save_context({"input": state["question"]}, {"output": answer})

    state["used_node"] = "sqlite"
    
    return state

# --- Router ---

def llm_router(state: ChatState) -> ChatState:
    # Use the router chain to get the decision
    # The router now explicitly receives 'question' and 'chat_history'
    choice = router_chain.invoke({
        "question": state["question"],
        "chat_history": state["chat_history"] # Pass the correctly prepared chat_history for context
    })
    
    # Validate the choice (assuming choice can be a dict from router_chain)
    if isinstance(choice, dict):
        choice = choice.get("choice", "sql") 
    
    if choice not in {"sql", "explain", "sqlite"}:
        choice = "sql"  
    
    state["used_node"] = choice
    
    return state

############################################################# Graph Setup ####################################################################

############# Graph #############

# Create the graph
graph = StateGraph(ChatState)

# Add nodes to the graph
graph.add_node("LLM Router", llm_router)
graph.add_node("SQL Retrieval Chain", run_full_chain)
graph.add_node("SQLite Retrieval Chain", run_sqlite_chain)
graph.add_node("Data Explanation Chain", run_data_explanation_chain)

# Add edges - start goes to router
graph.add_edge(START, "LLM Router")

# Add conditional edges based on the router's decision
graph.add_conditional_edges(
    "LLM Router", 
    lambda state: state["used_node"],
    {
        "sql": "SQL Retrieval Chain",
        "sqlite": "SQLite Retrieval Chain",
        "explain": "Data Explanation Chain"
    }
)

# End connections
graph.add_edge("SQL Retrieval Chain", END)
graph.add_edge("SQLite Retrieval Chain", END)
graph.add_edge("Data Explanation Chain", END)

# Compile the app
app = graph.compile()

##############################################################################
# START: AUTHENTICATION BLOCK [OPTIONAL]
##############################################################################

# Load configuration from config.yaml
try:
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("config.yaml not found. Please create it with your authentication details.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'], # Still assuming this is the correct path to credentials
    config['cookie_name'],
    config['cookie_key'],
    config['cookie_expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

# Handle authentication status
if authentication_status == False:
    st.error('Username/password is incorrect.')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password to access the chatbot.')
    st.stop()

# If authentication_status is True, the user is logged in.
if authentication_status:

    ########################################### Display App  #####################################

    ### Title 

    st.title("🧠 SQL Assistant")
            
    ### Warning Message before using 

    if "warning_dismissed" not in st.session_state:
        st.session_state.warning_dismissed = False

    if not st.session_state.warning_dismissed: # [OPTIONAL]
        with st.container(border=True):
            st.warning("⚠️ Important: Please Read Before Using")

            st.markdown("""
            - **Generative AI Disclaimer:** You are using a generative AI tool. **All outputs should be carefully checked for accuracy.**
            - **Data Timeliness:** The database contains historical data and may not be a reliable source for real-time, live information.
            - **Data Completeness:** The database may not contain all relevant data. Absence of data does not imply absence of events or facts.
            - **Data Security:** Do not share sensitive or confidential information with others who have no access rights.
            - **Feedback & Support:** For feedback or specific data requests, please reach out to [xxx@xxx.com](mailto:xxx@xxx.com)
            """)

            if st.button("I understand, close this message", key="dismiss_warning"):
                st.session_state.warning_dismissed = True
                #st.rerun()

    # The rest of the app loads only after the warning is dismissed
    if st.session_state.warning_dismissed: # [OPTIONAL]]
            # Streamlit UI starts here
        # ---------------------------
        with st.popover("💡 Show Example Prompts"):
                st.subheader("Example Prompts")                
                st.info("Prompt1")
                st.info("Prompt2")



        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)

        user_input_label = "Hello! I am the XXX Chatbot, ask a question about XXX:" #[Adjust to your needs]

        if user_input := st.chat_input(user_input_label):
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            st.session_state.chat_history.append(HumanMessage(content=user_input))

            # Prepare initial state for the graph
            initial_state = ChatState(
                question=user_input,
                memory=st.session_state.memory,
                chat_history=st.session_state.chat_history.copy(), # Pass a copy for immutability during graph execution
                answer=None,
                query=None,
                dataframe=None,
                used_node=None
            )

            final_state = None # Initialize outside try-except
            success = False

            with st.spinner("Generating and running query... 🍵"):
                success = False
                response = None
                for attempt in range(3):
                    try:
                        final_state = app.invoke(initial_state)
                        success = True
                        break
                    except APIConnectionError:
                        st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                        time.sleep(2)

            if success and final_state: 

                st.session_state.chat_history = final_state["chat_history"]
                st.session_state.memory = final_state["memory"]

                # Store query, dataframe, and used_node for sidebar and display
                st.session_state["last_query"] = final_state.get("query", "")
                st.session_state["last_dataframe"] = final_state.get("dataframe", pd.DataFrame())
                st.session_state["last_used_node"] = final_state.get("used_node", "")

                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(final_state["answer"])

                # 🆕 Save chat data to SharePoint CSV (completely hidden)
                def save_in_background():
                    try:
                        save_chat_data_to_sharepoint(
                            question=user_input,
                            query=final_state.get("query", ""),
                            used_node=final_state.get("used_node", ""),
                            answer=final_state.get("answer", "")
                        )
                    except:
                        pass

                # Run in background thread (completely invisible to user)
                threading.Thread(target=save_in_background, daemon=True).start()

            else:
                with st.chat_message("assistant", avatar="🧠"):
                    st.error("❌ Failed to get a response after multiple attempts. Please check your internet/API keys or try again later.")
                    # Clear previous query/dataframe if an error occurred
                    st.session_state["last_query"] = ""
                    st.session_state["last_dataframe"] = pd.DataFrame()
                    st.session_state["last_used_node"] = "" # Clear node info too

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

            if not st.session_state["last_dataframe"].empty:
                excel_bytes = to_excel_bytes(st.session_state["last_dataframe"])
                st.download_button(
                    label="⬇️ Download Last Query Result as Excel",
                    data=excel_bytes,
                    file_name="query_result.xlsx", # Changed from last_query_result.xlsx
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.write("No queries yet.")

        # Display list of tables
        st.write("### 🗃️ Available Tables")

        if st.session_state["table_names"]: # Check if table names are loaded
            with st.expander("Click to view tables"):
                for table in st.session_state["table_names"]:
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

            # Display the last used node
        if st.session_state["last_used_node"]:
            st.write("### ⚙️ Executed Node")
            st.info(f"The last request was handled by: **{st.session_state['last_used_node'].upper()}**")

        if st.button("🔄 Reset Chat Memory", type="secondary"): # Using 'secondary' for a distinct look
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state.chat_history = []

        # Sidebar - Reset session button
        if st.button("🔄 Reset Chat & Clear Cache"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_resource.clear()  
            st.rerun()