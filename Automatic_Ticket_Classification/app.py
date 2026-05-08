import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import (
    get_embeddings, get_faiss_index, get_svm_model, get_similar_docs, 
    get_answer, predict_department, read_pdf_data, split_data, push_to_faiss
)

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="AI Corporate Ticketing", page_icon="🏢", layout="wide")

# Initialize session states for tickets
for dept in ['HR_tickets', 'IT_tickets', 'Transport_tickets']:
    if dept not in st.session_state:
        st.session_state[dept] = []

st.title("🏢 Unified AI Ticketing & Policy Dashboard")
st.markdown("Navigate through the tabs below to manage data, train models, and interact with the AI.")

# ==========================================
# 2. CREATE UNIFIED TABS
# ==========================================
tab_chat, tab_upload, tab_train, tab_admin = st.tabs([
    "💬 1. Employee Portal (Chat)", 
    "📂 2. Upload Policies (Admin)", 
    "🧠 3. Train Model (Admin)", 
    "🎫 4. Pending Tickets (Admin)"
])

# ------------------------------------------
# TAB 1: EMPLOYEE PORTAL (CHAT & TICKETS)
# ------------------------------------------
with tab_chat:
    st.header("👋 How can I help you today?")
    
    # Load AI Assets
    embeddings = get_embeddings()
    faiss_index = get_faiss_index(embeddings)
    svm_model = get_svm_model()

    user_input = st.text_input("Ask a policy question or describe your issue:", placeholder="e.g., What is the monthly payslip policy?")

    if user_input:
        if not faiss_index:
            st.warning("⚠️ Policy Database not found. Please go to the 'Upload Policies' tab and upload your PDFs.")
        else:
            with st.spinner("Searching company policies..."):
                relavant_docs = get_similar_docs(faiss_index, user_input)
                response = get_answer(relavant_docs, user_input)
                
                st.info(f"🤖 **AI Answer:**\n\n{response}")
                
                st.markdown("---")
                st.markdown("### 🛠️ Need further assistance?")
                
                if st.button("Submit Support Ticket"):
                    if not svm_model:
                        st.error("⚠️ AI Classification Model not found. Please go to the 'Train Model' tab first.")
                    else:
                        department = predict_department(user_input, embeddings, svm_model)
                        
                        if department == "HR":
                            st.session_state['HR_tickets'].append(user_input)
                        elif department == "IT":
                            st.session_state['IT_tickets'].append(user_input)
                        else:
                            st.session_state['Transport_tickets'].append(user_input)
                            
                        st.success(f"✅ Ticket successfully created and routed to the **{department} Department**!")

# ------------------------------------------
# TAB 2: UPLOAD POLICIES (RAG)
# ------------------------------------------
with tab_upload:
    st.header("📂 Step 1: Upload Company Policies")
    st.write("Upload your HR, IT, and Transport policy PDFs here so the AI can read and learn them.")
    
    pdf = st.file_uploader('Upload PDF Document', type=["pdf"], key="pdf_uploader")

    if pdf is not None:
        if st.button("Process & Learn Document"):
            with st.status("Processing Document...", expanded=True) as status:
                st.write("📖 Reading PDF data...")
                text = read_pdf_data(pdf)
                
                st.write("✂️ Splitting data into chunks...")
                docs_chunks = split_data(text)
                
                st.write("🧠 Connecting to AI brain (Embeddings)...")
                emb = get_embeddings()
                
                st.write("💾 Saving to FAISS Vector Database...")
                push_to_faiss(docs_chunks, emb)
                
                status.update(label="Document Processing Complete!", state="complete", expanded=False)
                
            st.success("✅ Successfully stored! The AI can now answer questions based on this document.")
            st.cache_resource.clear() # Refresh memory

# ------------------------------------------
# TAB 3: TRAIN ML MODEL (SVM)
# ------------------------------------------
with tab_train:
    st.header("🧠 Step 2: Train Department Routing Model")
    st.write("Upload your historical ticket CSV (`Balanced_Ticket_Classification_Dataset.csv`) to train the AI where to send tickets.")
    
    data = st.file_uploader("Upload Ticket Dataset (CSV)", type="csv", key="csv_uploader")

    if data:
        df = pd.read_csv(data)
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Start Training Model"):
            with st.status("Training AI Model...", expanded=True) as status:
                st.write("⏳ Generating text embeddings... (This takes a moment)")
                emb = get_embeddings()
                
                # Assuming Columns are 'Ticket_Text' and 'Department'
                X_text = df['Ticket_Text'].tolist()
                y = df['Department'].tolist()
                X_embedded = emb.embed_documents(X_text)
                
                st.write("📊 Splitting Data...")
                X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.2, random_state=42)
                
                st.write("⚙️ Training SVM Classifier...")
                clf = SVC(kernel='linear')
                clf.fit(X_train, y_train)
                
                predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                
                st.write("💾 Saving Model to disk...")
                joblib.dump(clf, 'modelsvm.pk1')
                
                status.update(label=f"Training Complete! Accuracy: {acc*100:.2f}%", state="complete")
            
            st.success("✅ Model trained and saved! The app is now ready to route tickets in the Employee Portal.")
            st.cache_resource.clear() # Refresh memory

# ------------------------------------------
# TAB 4: PENDING TICKETS (ADMIN DASHBOARD)
# ------------------------------------------
with tab_admin:
    st.header("🎫 Pending Department Tickets")
    st.write("View all tickets submitted by employees, automatically sorted by the AI.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧑‍💼 HR Support")
        if not st.session_state['HR_tickets']:
            st.info("No pending tickets.")
        for i, ticket in enumerate(st.session_state['HR_tickets']):
            st.warning(f"**#{i+1}:** {ticket}")

    with col2:
        st.subheader("💻 IT Support")
        if not st.session_state['IT_tickets']:
            st.info("No pending tickets.")
        for i, ticket in enumerate(st.session_state['IT_tickets']):
            st.error(f"**#{i+1}:** {ticket}")

    with col3:
        st.subheader("🚗 Transportation")
        if not st.session_state['Transport_tickets']:
            st.info("No pending tickets.")
        for i, ticket in enumerate(st.session_state['Transport_tickets']):
            st.success(f"**#{i+1}:** {ticket}")