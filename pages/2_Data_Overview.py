
import os
import streamlit as st
import pandas as pd
import re 
import warnings
from sqlalchemy import text

from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase

import urllib
from sqlalchemy import create_engine, inspect

########################################### Chatbot #####################################################
st.set_page_config(layout="wide") # Use wide layout for a better chat experience

######################## Agent Streamlit Interface ########################

warnings.filterwarnings("ignore")

### Title 

### Load Environment
load_dotenv()

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
    prod_tables = ["XXX"]
    sales_tables = ["sales", "customers"]

    ######## Create databases 
        ## Sales Data

    sales_cols = ["XXX", "XX"]


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
        include_tables=prod_tables,
        schema="prod",
        sample_rows_in_table_info=2)
    
        ##  Filter cols
    prod_cols = ["XXX", ""]

    filtered_prod= filter_columns(
        schema_text = db_prod.get_table_info(),
        table_name = "XXX",                 
        allowed_columns = prod_cols)     # keep only these Columns

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
        'abc.[XXX]': 'SAP'
    }

    df['Source'] = df['Table Name'].map(source_mapping).fillna('Unknown')
    
    return df

if "database_overview" not in st.session_state:
    st.session_state["database_overview"] = get_colnames(db.get_table_info())

################ Overview
st.title("📊 Data Overview")

st.write("Here you can explore your customer data.")

sub_tabs = st.tabs(["Data Overview"])

with sub_tabs[0]:
    with st.expander("View Available Data"):
        st.write(st.session_state["database_overview"])

with sub_tabs[1]:
    st.subheader("XXX")

with sub_tabs[2]:
    st.subheader("XXX")
