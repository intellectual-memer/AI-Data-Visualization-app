import streamlit as st
import pandas as pd
import plotly.express as px
import sqlalchemy
import os
from pandasai import PandasAI
from pandasai.llm import OpenAI
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize global variables
#global df_global
#global pandas_ai
df_global = pd.DataFrame()
openai_api_key = os.getenv('OPENAI_API_KEY')  # Ensure OpenAI API key is set
postgres_uri = os.getenv('POSTGRES_DATABASE_URI')  # Ensure PostgreSQL URI is set
pandas_ai = None

# Initialize OpenAI for PandasAI
if openai_api_key:
    llm = OpenAI(api_key=openai_api_key)
    pandas_ai = PandasAI(llm)
else:
    st.error("OpenAI API key not found! Please set it in your environment variables.")

# Streamlit App Title
st.title("Streamlit Data Science Web App")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an option:", ["Upload CSV", "Query Database", "Kaggle Datasets"])

# Shared AI Question Prompt Functionality
def ai_question_prompt(unique_key):
    global pandas_ai, df_global
    #if "df_global" not in st.session_state or st.session_state["df_global"].empty:
    if df_global.empty:
        st.warning("No data available. Please upload or query data first.")
        return
    if not pandas_ai:
        st.warning("PandasAI is not initialized. Please try again.")
        return

    st.subheader("Ask Questions About Your Data")
    question = st.text_input("Enter your question about the data:", key=unique_key)
    if st.button("Ask", key=f"{unique_key}_button"):
        try:
            response = pandas_ai.run(df_global, question)
            st.success(f"Response: {response}")
        except Exception as e:
            st.error(f"Error processing question: {e}")

# Shared Visualization Functionality
def generate_visualizations():
    #if "df_global" in st.session_state and not st.session_state["df_global"].empty:
    if not df_global.empty:
        st.subheader("Generate Visualizations")
        x_axis = st.selectbox("Select X-axis:", options=df_global.columns)
        y_axis = st.selectbox("Select Y-axis:", options=df_global.columns)
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Scatter Plot", "Bar Chart", "Line Graph", "Box Plot", "Histogram", "Heatmap"],
        )

        if st.button("Generate Chart"):
            try:
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df_global, x=x_axis, y=y_axis)
                elif chart_type == "Bar Chart":
                    fig = px.bar(df_global, x=x_axis, y=y_axis)
                elif chart_type == "Line Graph":
                    fig = px.line(df_global, x=x_axis, y=y_axis)
                elif chart_type == "Box Plot":
                    fig = px.box(df_global, x=x_axis, y=y_axis)
                elif chart_type == "Histogram":
                    fig = px.histogram(df_global, x=x_axis)
                elif chart_type == "Heatmap":
                    fig = px.density_heatmap(df_global, x=x_axis, y=y_axis)

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating chart: {e}")

# ---- CSV Upload ----
if options == "Upload CSV":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            #global df_global
            df_global = pd.read_csv(uploaded_file)
            st.success(f"CSV file uploaded successfully! Shape: {df_global.shape}")
            st.dataframe(df_global.head())

            # Initialize PandasAI
            #global pandas_ai
            if not df_global.empty and openai_api_key:
                pandas_ai = PandasAI(llm)
                st.info("PandasAI initialized for data insights.")
        except Exception as e:
            st.error(f"Error uploading CSV: {e}")

    # Visualization
    generate_visualizations()

    # AI Question Prompt
    ai_question_prompt(unique_key="ai_question_input_csv")

# ---- Query Database ----
elif options == "Query Database":
    st.header("Query PostgreSQL Database")
    
    # Use session state to maintain the query value
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # Update the session state with the current input
    query = st.text_area("Enter your SQL query:", value=st.session_state.query, key="query_input")

    if st.button("Run Query"):
        try:
            if not postgres_uri:
                st.error("PostgreSQL URI not configured! Set it in environment variables.")
            else:
                engine = sqlalchemy.create_engine(postgres_uri)
                df_global = pd.read_sql(query, engine)
                st.session_state["query"] = query  # Save the query
                st.session_state["df_global"] = df_global  # Save the DataFrame
                st.success("Query executed successfully!")
                #st.dataframe(df_global.head())
                st.session_state["df_global_head"] = df_global.head()  # Save the DataFrame
                st.session_state["df_global_head"]

                # # Store the query in session state
                # st.session_state.query = query  # Ensure the query is stored after execution

                # Reinitialize PandasAI
                if not df_global.empty and openai_api_key:
                    pandas_ai = PandasAI(llm)
                    st.info("PandasAI initialized for data insights.")
                    
        except Exception as e:
            st.error(f"Error executing query: {e}")

        # Check if data exists in session_state
    if "df_global" in st.session_state:
        df_global = st.session_state["df_global"]  # Retrieve saved DataFrame

        # Visualization
        generate_visualizations()
        # AI Question Prompt
        ai_question_prompt(unique_key="ai_question_input_db")

# # ---- Kaggle Datasets ----
# elif options == "Kaggle Datasets":
#     st.header("Fetch Kaggle Datasets")
#     try:
#         kaggle_api = KaggleApi()
#         kaggle_api.authenticate()

#         datasets = kaggle_api.dataset_list()
#         st.success("Kaggle datasets fetched successfully!")
#         for dataset in datasets[:10]:
#             st.markdown(f"[{dataset.title}]({dataset.url})")
        
#         dataset_url = st.text_input("Enter the Kaggle dataset URL (e.g., https://www.kaggle.com/datasets/username/dataset-name):")
#         if st.button("Download Kaggle Dataset"):
#             if dataset_url:
#                 try:
#                     dataset_name = dataset_url.split("/")[-2] + "/" + dataset_url.split("/")[-1]
#                     kaggle_api.dataset_download_files(dataset_name, path="./", unzip=True)
                    
#                     files = [f for f in os.listdir("./") if f.endswith(".csv")]
#                     if files:
#                         #global df_global
#                         df_global = pd.read_csv(files[0])
#                         st.success(f"Kaggle dataset downloaded successfully! Shape: {df_global.shape}")
#                         st.dataframe(df_global.head())

#                         # Initialize PandasAI
#                         #global pandas_ai
#                         if not df_global.empty and openai_api_key:
#                             pandas_ai = PandasAI(llm)
#                             st.info("PandasAI initialized for data insights.")
#                     else:
#                         st.error("No CSV file found in the downloaded dataset.")
#                 except Exception as e:
#                     st.error(f"Error downloading Kaggle dataset: {e}")
#             else:
#                 st.warning("Please enter a valid Kaggle dataset URL.")
#     except Exception as e:
#         st.error(f"Error fetching Kaggle datasets: {e}")

#     # Visualization
#     generate_visualizations()

#     # AI Question Prompt
#     ai_question_prompt(unique_key="ai_question_input_kaggle")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("AI Data Visualization App")
st.sidebar.text("Powered by PandasAI, OpenAI, and Plotly")