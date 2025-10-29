import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================================
# 🎨 Streamlit Page Config & Styling
# ===========================================
st.set_page_config(
    page_title="SmartMail Classifier",
    page_icon="📧",
    layout="wide",
)

st.markdown("""
    <style>
       # [data-testid="stSidebar"] {display: none;}
        body {
            background: linear-gradient(120deg, #c9ffbf 0%, #ffafbd 100%);
        }
        .main-title {
            font-size: 45px;
            font-weight: 800;
            text-align: center;
            color: #333;
            margin-bottom: 5px;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #2e86de;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-size: 1em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #1b4f72;
            color: #f0f0f0;
        }
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===========================================
# 🧠 Title
# ===========================================
st.markdown("<div class='main-title'>📧 SmartMail Spam Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Classify emails as <b>Spam</b> or <b>Not Spam</b> using AI models (KNN & SVM)</div>", unsafe_allow_html=True)

# ===========================================
# 📂 Load Dataset (from local file)
# ===========================================
@st.cache_data
def load_data():
    return pd.read_csv("emails.csv")

df = load_data()

# ===========================================
# 🧹 Data Preprocessing
# ===========================================
if 'Prediction' in df.columns:
    y = df['Prediction']
else:
    st.error("❌ No 'Prediction' column found in dataset.")
    st.stop()

# Detect if dataset is text-based or numeric
if 'Email' in df.columns:  # text dataset
    X = df['Email']
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
else:  # numeric dataset
    X = df.select_dtypes(include=['number'])
    X_vec = X  # no vectorization needed

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# ===========================================
# 🤖 Train Models
# ===========================================
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# ===========================================
# ⚖️ Model Comparison
# ===========================================
st.markdown("### ⚙️ Model Performance Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("KNN Accuracy", f"{acc_knn*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("SVM Accuracy", f"{acc_svm*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

if acc_knn > acc_svm:
    st.success(f"✅ KNN performed better ({acc_knn*100:.2f}% vs {acc_svm*100:.2f}%)")
elif acc_svm > acc_knn:
    st.success(f"✅ SVM performed better ({acc_svm*100:.2f}% vs {acc_knn*100:.2f}%)")
else:
    st.info("🤝 Both models performed equally well!")

# ===========================================
# 📉 Confusion Matrices
# ===========================================
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)

st.markdown("### 🧩 Confusion Matrices")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("KNN Confusion Matrix")
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("SVM Confusion Matrix")
st.pyplot(fig)

# ===========================================
# ✉️ Predict New Email
# ===========================================
st.markdown("### 🧠 Try Your Own Email")
user_input = st.text_area("✉️ Enter email text below:", height=120, placeholder="Type or paste an email here...")

if st.button("🔍 Classify Email"):
    if 'Email' not in df.columns:
        st.info("⚙️ This dataset is numeric — text input not supported in this mode.")
    elif user_input.strip() == "":
        st.warning("Please enter an email message first.")
    else:
        new_vec = vectorizer.transform([user_input])
        knn_pred = knn.predict(new_vec)[0]
        svm_pred = svm.predict(new_vec)[0]

        col1, col2 = st.columns(2)
        col1.markdown("#### 🧩 KNN Prediction")
        col1.info("📩 SPAM" if knn_pred == 1 else "✅ NOT SPAM")

        col2.markdown("#### ⚙️ SVM Prediction")
        col2.info("📩 SPAM" if svm_pred == 1 else "✅ NOT SPAM")

        if (knn_pred + svm_pred) / 2 >= 0.5:
            st.error("⚠️ This email is likely **SPAM!** Be cautious 🚫")
        else:
            st.success("💌 This email seems **safe** and not spam ✅")

# ===========================================
# 🪄 Footer
# ===========================================
st.markdown("""
---
<div style='text-align:center; color:gray;'>
Made with ❤️ using <b>Streamlit</b> | SmartMail © 2025
</div>
""", unsafe_allow_html=True)
