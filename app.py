import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

st.set_page_config(page_title="AI Sales Intelligence", layout="wide", page_icon="🤖")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def train_models(df, tenure_col, charges_col, contract_col, churn_col):
    data = df.copy()
    data = data.loc[:, ~data.columns.duplicated()]

    data[tenure_col] = pd.to_numeric(data[tenure_col], errors='coerce')
    data[charges_col] = pd.to_numeric(data[charges_col], errors='coerce')
    data.dropna(subset=[tenure_col, charges_col], inplace=True)

    keep_cols = [tenure_col, charges_col, contract_col, churn_col]
    id_like_cols = [col for col in data.columns if ('id' in col.lower()) and col not in keep_cols]
    data.drop(columns=id_like_cols, inplace=True, errors='ignore')

    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col].astype(str))

    churn_vals = data[churn_col].unique()
    if len(churn_vals) > 2:
        threshold = data[churn_col].median()
        data[churn_col] = (data[churn_col] > threshold).astype(int)

    y = data[churn_col]
    X = data.drop(columns=[churn_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    churn_model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    churn_model.fit(X_train, y_train)
    data['churn_probability'] = churn_model.predict_proba(X)[:, 1]
    data['churn_risk'] = data['churn_probability'].apply(
        lambda x: 'HIGH' if x > 0.7 else ('MEDIUM' if x > 0.4 else 'LOW')
    )

    data['lead_score'] = (
        (data[tenure_col] / data[tenure_col].max()) * 40 +
        (data[charges_col] / data[charges_col].max()) * 30 +
        (data[contract_col] / data[contract_col].max()) * 30
    ).clip(0, 100)

    X_lead = data.drop(columns=[churn_col, 'lead_score', 'churn_probability', 'churn_risk'])
    y_lead = data['lead_score']
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lead, y_lead, test_size=0.2, random_state=42)
    lead_model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=1)
    lead_model.fit(X_train_l, y_train_l)
    data['lead_score_predicted'] = lead_model.predict(X_lead)

    accuracy = accuracy_score(y_test, churn_model.predict(X_test))
    st.session_state['accuracy'] = round(accuracy * 100, 1)

    return data

def build_vector_db(df, tenure_col, charges_col, churn_col):
    model_st = get_embedding_model()
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="customers")
    except:
        pass
    collection = chroma_client.create_collection(name="customers")
    def to_text(row):
        churn_status = "churned" if row[churn_col] == 1 else "stayed as customer"
        return (
            f"Customer with {row[tenure_col]} months tenure, "
            f"${row[charges_col]:.0f} monthly charges, "
            f"lead score {row['lead_score']:.0f}/100, "
            f"and {churn_status}."
        )
    texts = [to_text(row) for _, row in df.iterrows()]
    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model_st.encode(batch_texts).tolist()
        batch_ids = [str(j) for j in range(i, i+len(batch_texts))]
        collection.add(documents=batch_texts, embeddings=batch_embeddings, ids=batch_ids)
    return collection, texts, model_st

def find_similar(index, model_st, collection, texts, top_n=3):
    query_embedding = model_st.encode([texts[index]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_n+1)
    similar = []
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        if results['ids'][0][i] == str(index):
            continue
        similar.append({"text": doc, "similarity": round((1 - distance) * 100, 1)})
    return similar[:top_n]

def generate_email(row, similar, groq_key, tenure_col, charges_col):
    similar_text = "\n".join([f"- {s['text']}" for s in similar])
    prompt = f"""
You are a helpful sales assistant. Write a short, warm, personalised email to a customer with these details:
- Tenure: {row[tenure_col]} months
- Monthly charges: ${row[charges_col]}
- Churn risk: {row['churn_probability']*100:.0f}%
- Lead score: {row['lead_score']:.0f}/100

Here are 3 similar past customers:
{similar_text}

If churn risk is high (>70%), focus on retention.
If lead score is high (>75%), focus on upselling.
Keep it under 100 words. Sound human, not robotic.
"""
    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ─── UI STARTS HERE ───────────────────────────────────────────────

st.title(" AI Sales Intelligence Platform")
st.markdown("*Upload any customer dataset — get instant churn prediction, lead scoring, and AI-generated emails*")
st.markdown("---")

with st.sidebar:
    st.header(" Settings")
    groq_key = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload your CSV")
    st.markdown("2. Map your columns")
    st.markdown("3. Click Analyse")
    st.markdown("4. Explore insights")
    st.markdown("5. Generate emails")

uploaded_file = st.file_uploader(
    " Upload your customer CSV file",
    type=["csv"],
    help="Any CSV with customer data — tenure, charges, churn column required"
)

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.success(f" Loaded {len(raw_df):,} customers with {len(raw_df.columns)} columns")

    with st.expander(" Preview your data", expanded=False):
        st.dataframe(raw_df.head())

    st.markdown("---")
    st.subheader(" Map Your Columns")
    st.markdown("Tell us which columns represent each field:")

    cols = raw_df.columns.tolist()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tenure_col = st.selectbox(" Tenure / Months active", cols, help="How long the customer has been with you")
    with col2:
        charges_col = st.selectbox(" Monthly charges / Spend", cols, help="How much they pay per month")
    with col3:
        contract_col = st.selectbox(" Contract / Plan type", cols, help="Type of contract or plan")
    with col4:
        churn_col = st.selectbox(" Churn column", cols, help="Did the customer leave? (0/1 or Yes/No)")

    st.markdown("---")

    if st.button(" Analyse My Customers", type="primary", use_container_width=True):
        with st.spinner("Training churn model..."):
            df = train_models(raw_df.copy(), tenure_col, charges_col, contract_col, churn_col)
        with st.spinner("Building vector database..."):
            collection, texts, model_st = build_vector_db(df, tenure_col, charges_col, churn_col)
        st.session_state['df'] = df
        st.session_state['collection'] = collection
        st.session_state['texts'] = texts
        st.session_state['model_st'] = model_st
        st.session_state['tenure_col'] = tenure_col
        st.session_state['charges_col'] = charges_col
        st.session_state['churn_col'] = churn_col
        st.success(" Analysis complete!")

if 'df' in st.session_state:
    df = st.session_state['df']
    collection = st.session_state['collection']
    texts = st.session_state['texts']
    model_st = st.session_state['model_st']
    tenure_col = st.session_state['tenure_col']
    charges_col = st.session_state['charges_col']
    churn_col = st.session_state['churn_col']

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("🔴 High Risk", f"{len(df[df['churn_risk']=='HIGH']):,}")
    c3.metric("🟡 Medium Risk", f"{len(df[df['churn_risk']=='MEDIUM']):,}")
    c4.metric("Avg Lead Score", f"{df['lead_score'].mean():.1f}/100")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(" Top 10 Leads")
        top10 = df.nlargest(10, 'lead_score').reset_index()
        fig = px.bar(top10, x=top10.index, y='lead_score', color='churn_risk',
                     color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
                     labels={'y': 'Lead Score', 'x': 'Customer'})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader(" Churn Risk Distribution")
        risk_counts = df['churn_risk'].value_counts().reset_index()
        fig2 = px.pie(risk_counts, values='count', names='churn_risk',
                      color='churn_risk',
                      color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader(" Customer Table")

    risk_filter = st.multiselect("Filter by risk", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    filtered = df[df['churn_risk'].isin(risk_filter)].reset_index(drop=False)

    def risk_badge(r):
        return '🔴 HIGH' if r == 'HIGH' else ('🟡 MEDIUM' if r == 'MEDIUM' else '🟢 LOW')

    display = filtered[[tenure_col, charges_col, 'lead_score', 'churn_probability', 'churn_risk']].copy()
    display['churn_risk'] = display['churn_risk'].apply(risk_badge)
    display['churn_probability'] = (display['churn_probability'] * 100).round(1).astype(str) + '%'
    display['lead_score'] = display['lead_score'].round(1)
    st.dataframe(display, use_container_width=True)

    st.markdown("---")
    st.subheader(" Customer Deep Dive")

    customer_index = st.selectbox("Select a customer", filtered['index'].tolist())

    if customer_index is not None:
        row = df.iloc[customer_index]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tenure", f"{row[tenure_col]} months")
        m2.metric("Monthly Charges", f"${row[charges_col]:.0f}")
        m3.metric("Lead Score", f"{row['lead_score']:.0f}/100")
        m4.metric("Churn Risk", f"{row['churn_probability']*100:.0f}%")

        st.markdown("** Similar Past Customers:**")
        similar = find_similar(customer_index, model_st, collection, texts)
        for s in similar:
            st.info(f"**{s['similarity']}% match** — {s['text']}")

        st.markdown("---")
        if groq_key:
            if st.button(" Generate Personalised Email", type="primary"):
                with st.spinner("Generating email..."):
                    email = generate_email(row, similar, groq_key, tenure_col, charges_col)
                    st.success(email)
        else:
            st.warning(" Enter your Groq API key in the sidebar to generate emails.")

else:
    st.info("👆 Upload a CSV file above to get started.")
    st.markdown("---")
    st.subheader(" What this platform does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("###  Churn Prediction\nIdentifies customers likely to leave using Random Forest ML model")
    with col2:
        st.markdown("###  Lead Scoring\nScores every customer 0-100 based on loyalty, spend and commitment")
    with col3:
        st.markdown("###  AI Emails\nGenerates personalised emails using similar customer history as context")