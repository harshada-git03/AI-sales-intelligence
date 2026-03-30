import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="AI Sales Intelligence", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv('data/telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
    churn_model.fit(X_train, y_train)
    df['churn_probability'] = churn_model.predict_proba(X)[:, 1]
    df['churn_risk'] = df['churn_probability'].apply(lambda x: 'HIGH' if x > 0.7 else ('MEDIUM' if x > 0.4 else 'LOW'))
    df['lead_score'] = (
        (df['tenure'] / df['tenure'].max()) * 40 +
        (df['MonthlyCharges'] / df['MonthlyCharges'].max()) * 30 +
        df['Contract'].map({0: 0, 1: 15, 2: 30})
    )
    X_lead = df.drop(columns=['Churn', 'lead_score', 'churn_probability', 'churn_risk'])
    y_lead = df['lead_score']
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lead, y_lead, test_size=0.2, random_state=42)
    lead_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    lead_model.fit(X_train_l, y_train_l)
    df['lead_score_predicted'] = lead_model.predict(X_lead)
    return df

@st.cache_resource
def load_vector_db(df):
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="customers")
    except:
        pass
    collection = chroma_client.create_collection(name="customers")
    def customer_to_text(row):
        churn_status = "churned" if row['Churn'] == 1 else "stayed as customer"
        return (
            f"Customer with {row['tenure']} months tenure, "
            f"${row['MonthlyCharges']:.0f} monthly charges, "
            f"lead score {row['lead_score']:.0f}/100, "
            f"and {churn_status}."
        )
    texts = [customer_to_text(row) for _, row in df.iterrows()]
    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model_st.encode(batch_texts).tolist()
        batch_ids = [str(j) for j in range(i, i+len(batch_texts))]
        collection.add(documents=batch_texts, embeddings=batch_embeddings, ids=batch_ids)
    return model_st, collection, texts

def find_similar(index, model_st, collection, texts, top_n=3):
    query_embedding = model_st.encode([texts[index]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_n+1)
    similar = []
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        if results['ids'][0][i] == str(index):
            continue
        similar.append({"text": doc, "similarity": round((1 - distance) * 100, 1)})
    return similar[:top_n]

def generate_email(row, similar, groq_client):
    similar_text = "\n".join([f"- {s['text']}" for s in similar])
    prompt = f"""
You are a helpful sales assistant. Write a short, warm, personalised email to a customer with these details:
- Tenure: {row['tenure']} months
- Monthly charges: ${row['MonthlyCharges']}
- Churn risk: {row['churn_probability']*100:.0f}%
- Lead score: {row['lead_score']:.0f}/100

Here are 3 similar past customers:
{similar_text}

If churn risk is high (>70%), focus on retention.
If lead score is high (>75%), focus on upselling.
Keep it under 100 words. Sound human, not robotic.
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title(" AI Sales Intelligence Dashboard")
st.markdown("---")

with st.spinner("Loading models and data..."):
    df = load_and_train()

with st.spinner("Building vector database..."):
    model_st, collection, texts = load_vector_db(df)

with st.sidebar:
    st.header("🔑 Settings")
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    st.markdown("---")
    st.header(" Filters")
    risk_filter = st.multiselect("Churn Risk", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    contract_filter = st.multiselect("Contract Type", [0, 1, 2], default=[0, 1, 2], format_func=lambda x: {0: "Month-to-month", 1: "One year", 2: "Two year"}[x])

filtered_df = df[df['churn_risk'].isin(risk_filter) & df['Contract'].isin(contract_filter)].reset_index(drop=False)

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(filtered_df))
col2.metric("High Risk", len(filtered_df[filtered_df['churn_risk'] == 'HIGH']))
col3.metric("Avg Lead Score", f"{filtered_df['lead_score'].mean():.1f}/100")

st.markdown("---")

st.subheader(" Top 10 Leads by Score")
top10 = filtered_df.nlargest(10, 'lead_score')[['index', 'lead_score', 'churn_risk', 'tenure', 'MonthlyCharges']]
fig = px.bar(top10, x='index', y='lead_score', color='churn_risk',
             color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'},
             labels={'index': 'Customer Index', 'lead_score': 'Lead Score'})
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("👥 Customer Table")
def risk_badge(risk):
    if risk == 'HIGH': return '🔴 HIGH'
    if risk == 'MEDIUM': return '🟡 MEDIUM'
    return '🟢 LOW'

display_df = filtered_df[['index', 'tenure', 'MonthlyCharges', 'lead_score', 'churn_probability', 'churn_risk']].copy()
display_df['churn_risk'] = display_df['churn_risk'].apply(risk_badge)
display_df['churn_probability'] = (display_df['churn_probability'] * 100).round(1).astype(str) + '%'
display_df['lead_score'] = display_df['lead_score'].round(1)
st.dataframe(display_df, use_container_width=True)

st.markdown("---")

st.subheader("🔎 Customer Deep Dive")
customer_index = st.selectbox("Select a customer", filtered_df['index'].tolist())

if customer_index is not None:
    row = df.iloc[customer_index]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tenure", f"{row['tenure']} months")
    c2.metric("Monthly Charges", f"${row['MonthlyCharges']:.0f}")
    c3.metric("Lead Score", f"{row['lead_score']:.0f}/100")
    c4.metric("Churn Risk", f"{row['churn_probability']*100:.0f}%")

    st.markdown("**Similar Past Customers:**")
    similar = find_similar(customer_index, model_st, collection, texts)
    for s in similar:
        st.info(f"**{s['similarity']}% match** — {s['text']}")

    st.markdown("---")
    if groq_key:
        if st.button("✉️ Generate Personalised Email"):
            with st.spinner("Generating email..."):
                groq_client = Groq(api_key=groq_key)
                email = generate_email(row, similar, groq_client)
                st.success(email)
    else:
        st.warning("Enter your Groq API key in the sidebar to generate emails.")