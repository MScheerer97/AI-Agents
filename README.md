# AI-Agents
*SQL Agent - Multi-Tool Agent* with Langchain &amp; Langgraph

## Project Overview
This project was created to automate inquiries & support ad-hoc data analytics.
By accessing your SQL database, the chatbot gives accurate responses & provides the raw data in an excel file ready to download.

### Requirements
- Azure OpenAI API Key
- SQL database with user id & password access
- Docker & Cloud Environment (in my case: Cloud Foundry)

## Project Workflow
<img src="https://github.com/MScheerer97/AI-Agents/blob/main/images/workflow.PNG" alt="Workflow" width="350"/>

## How to use the template
Here I will outline how anyone can implement the solution.

### Adjustments Needed 

### 1. .env file
Here you need to insert your OpenAI & SQL credentials.

### 2. Chatbot Code 
#### 2.1 Database 
Here you have to adjust the database setup with your table & column names. 
Note: in my case I use MSSQL, but works with any SQL database. 

#### 2.2 Prompt Template
Here you need to adjust many things:
- Ensure to explain the business background and provide insider terminology used for the different tables & columns. 
- Provide a long an detailed list with special instructions like when to use which column (e.g. when you have two columns for revenue with different currencies...)
- Also tell the LLM how the tables are related and provide join instructions
- Provide *many example prompts with the correct SQL query*, especially for frequent & complex queries.
- Adjust answer prompt accordingly

### 3. odbc.ini & odbcinst.ini
Also adjust your credentials here. 
This basically ensures the SQL driver installation in the container runs smoothly.
If the your driver is different, adjust.

### 3. Docker-compose yaml 
Only needed to adjust the name of the Docker Image.

### 4. Dockerfile
If your driver is different, adjust driver installation and change the file name at the end to match your streamlit app file name.

### 5. manifest.yaml (OPTIONAL: Cloud Foundry)
Adjust again all the credentials and add the Docker image registry link.







