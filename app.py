# UmeedRise – AI-Driven Student Dropout Prediction & Counseling System
# Streamlit app: production-ready, robust ML pipeline, SHAP explanations, Plotly dashboard, premium UI
# Author: GitHub Copilot
# Compatible with Streamlit Cloud

import io
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target

# Try optional libraries
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

APP_NAME = "UmeedRise – AI-Driven Student Dropout Prediction & Counseling System"

# ============================
# UI Styling
# ============================

PRIMARY_GRADIENT = """
background: linear-gradient(135deg, #6a73ff 0%, #8d7bff 40%, #b894ff 100%);
"""

CARD_STYLE = """
background-color: #ffffff;
border-radius: 18px;
box-shadow: 0 10px 30px rgba(0,0,0,0.10);
padding: 18px 20px;
border: 1px solid rgba(136, 136, 136, 0.15);
"""

CHIP_STYLE = """
display: inline-block;
padding: 5px 12px;
border-radius: 999px;
background: #f0f0ff;
color: #4c4cff;
font-weight: 600;
border: 1px solid rgba(120,120,255,0.3);
margin-right: 6px;
"""

RISK_COLORS = {
    "Low": "#34c759",      # green
    "Medium": "#ffcc00",   # yellow
    "High": "#ff3b30",     # red
}

def inject_global_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            {PRIMARY_GRADIENT}
            min-height: 100vh;
            background-attachment: fixed;
        }}
        section[data-testid="stSidebar"] > div {{
            {PRIMARY_GRADIENT}
            color: white !important;
        }}
        .sidebar-title {{
            font-weight: 800;
            font-size: 22px;
            margin-bottom: 8px;
            color: white;
        }}
        .app-title {{
            font-size: 34px;
            font-weight: 800;
            color: white;
            letter-spacing: 0.2px;
        }}
        .app-subtitle {{
            font-size: 16px;
            opacity: 0.95;
            color: white;
            font-weight: 500;
        }}
        .card {{
            {CARD_STYLE}
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(0,0,0,0.14);
        }}
        .chip {{
            {CHIP_STYLE}
        }}
        .metric-good {{ color: #34c759; font-weight: 700; }}
        .metric-mid {{ color: #ffcc00; font-weight: 700; }}
        .metric-bad {{ color: #ff3b30; font-weight: 700; }}
        .risk-low {{ color: #34c759; font-weight: 700; }}
        .risk-medium {{ color: #ffcc00; font-weight: 700; }}
        .risk-high {{ color: #ff3b30; font-weight: 700; }}
        .caption {{ font-size: 13px; opacity: 0.85; }}
        .small-note {{ font-size: 12px; opacity: 0.75; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ============================
# Core Functions
# ============================

def load_data(file: io.BytesIO) -> Optional[pd.DataFrame]:
    """
    Load CSV file robustly with encoding fallbacks.
    """
    if file is None:
        return None
    try:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding=enc)
                return df
            except Exception:
                continue
        file.seek(0)
        df = pd.read_csv(file, engine="python")
        return df
    except Exception as e:
        st.warning(f"Could not read CSV. Please upload a valid CSV. Details: {e}")
        return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect target, student ID, numeric and categorical features.
    """
    if df is None or df.empty:
        return {"target": None, "student_id": None, "numeric": [], "categorical": []}

    cols = df.columns.tolist()
    cols_lower = [str(c).lower() for c in cols]

    # Target detection
    target_candidates = ["target", "dropout", "label", "risk", "outcome", "status", "y", "class"]
    target = None
    for tc in target_candidates:
        for i, c in enumerate(cols_lower):
            if tc == c or tc in c:
                target = cols[i]
                break
        if target:
            break

    # Student ID detection
    id_candidates = ["student_id", "id", "student", "roll", "rollno", "roll_no", "admission", "enrollment"]
    student_id = None
    for ic in id_candidates:
        for i, c in enumerate(cols_lower):
            if ic == c or ic in c:
                student_id = cols[i]
                break
        if student_id:
            break

    # Types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Exclude specials
    for special in [target, student_id]:
        if special in numeric_cols:
            numeric_cols.remove(special)
        if special in categorical_cols:
            categorical_cols.remove(special)

    # Heuristic fallback: if no target, look for binary-like columns
    if target is None:
        for col in cols:
            unique_vals = pd.Series(df[col]).dropna().unique()
            if len(unique_vals) == 2 and col != student_id:
                target = col
                break

    return {"target": target, "student_id": student_id, "numeric": numeric_cols, "categorical": categorical_cols}


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessing:
    - Numeric: median impute + StandardScaler
    - Categorical: constant impute + OneHotEncoder (sparse_output=False)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def train_model(
    df: pd.DataFrame,
    target: str,
    student_id: Optional[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train classifier with XGBoost primary and RandomForest fallback.
    Returns pipeline, processed X_train/X_test, y_train/y_test, and transformed feature names.
    """
    if target is None or target not in df.columns:
        raise ValueError("Target column missing or not found.")

    # Feature columns
    feature_cols = [c for c in df.columns if c != target and c != student_id]
    if not numeric_cols and not categorical_cols:
        # Auto split based on dtypes on the chosen features
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X = df[feature_cols].copy()
    y_raw = df[target].copy()

    # Encode target
    y_type = type_of_target(y_raw)
    if y_type in ["binary", "multiclass"]:
        if not np.issubdtype(y_raw.dtype, np.number):
            y, _ = pd.factorize(y_raw)
        else:
            y = y_raw.astype(int)
    elif y_type == "continuous":
        st.warning("Detected continuous target; converting to binary using median threshold.")
        thresh = pd.to_numeric(y_raw, errors="coerce").median()
        y = (pd.to_numeric(y_raw, errors="coerce") >= thresh).astype(int)
    else:
        y = pd.factorize(y_raw)[0]

    # Stratified split if possible
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Classifier choice
    if XGB_AVAILABLE:
        objective = "binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob"
        classifier = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective=objective,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", classifier)])
    model.fit(X_train, y_train)

    # Feature names post-transform
    feature_names = []
    try:
        preprocessor.fit(X_train)
        num_features = numeric_cols
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
        except Exception:
            cat_features = categorical_cols
        feature_names = num_features + cat_features
    except Exception:
        try:
            n_out = model.named_steps["preprocessor"].transform(X_train).shape[1]
            feature_names = [f"f_{i}" for i in range(n_out)]
        except Exception:
            feature_names = []

    X_train_processed = model.named_steps["preprocessor"].transform(X_train)
    X_test_processed = model.named_steps["preprocessor"].transform(X_test)

    # Cache helpful info
    st.session_state["feature_cols"] = feature_cols
    st.session_state["numeric_cols"] = numeric_cols
    st.session_state["categorical_cols"] = categorical_cols

    return model, X_train_processed, X_test_processed, y_train, y_test, feature_names


def evaluate(model: Pipeline, X_test_processed: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance with robust handling for binary/multiclass.
    """
    y_pred = model.named_steps["clf"].predict(X_test_processed)
    y_proba = None
    try:
        y_proba = model.named_steps["clf"].predict_proba(X_test_processed)
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred))
    }
    average_strategy = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    metrics["precision"] = float(precision_score(y_test, y_pred, average=average_strategy, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, average=average_strategy, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, average=average_strategy, zero_division=0))

    try:
        if y_proba is not None:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            metrics["roc_auc"] = float(auc)
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def risk_from_proba(p: float, low_thr: float = 0.33, high_thr: float = 0.66) -> str:
    """
    Map probability to Low/Medium/High risk.
    """
    if p < low_thr:
        return "Low"
    elif p < high_thr:
        return "Medium"
    else:
        return "High"


def predict(
    model: Pipeline,
    df_input: pd.DataFrame,
    feature_cols: List[str],
    low_thr: float = 0.33,
    high_thr: float = 0.66
) -> pd.DataFrame:
    """
    Predict dropout risk for new data with colored risk bands and scores.
    """
    # Align columns
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = np.nan

    X_new = df_input[feature_cols].copy()

    try:
        proba = model.predict_proba(X_new)
        if proba.shape[1] == 2:
            risk_p = proba[:, 1]
        else:
            risk_p = proba.max(axis=1)
    except Exception:
        preds = model.predict(X_new)
        # Fallback: treat positive class as risk
        risk_p = (preds == np.max(preds)).astype(float)

    y_pred = model.predict(X_new)
    risks = [risk_from_proba(p, low_thr, high_thr) for p in risk_p]

    result = df_input.copy()
    result["predicted_class"] = y_pred
    result["dropout_risk_score"] = np.round(risk_p, 4)
    result["dropout_risk_band"] = risks

    return result


def explain_with_shap(model: Pipeline, X_processed: np.ndarray, feature_names: List[str]):
    """
    Initialize SHAP explainer and values. TreeExplainer preferred.
    """
    if not SHAP_AVAILABLE:
        st.warning("SHAP not available. Include 'shap' in requirements.")
        return None, None

    clf = model.named_steps["clf"]
    explainer = None
    shap_values = None

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_processed)
    except Exception:
        try:
            explainer = shap.Explainer(clf.predict, model.named_steps["preprocessor"].transform)
            shap_values = explainer(X_processed)
        except Exception as e:
            st.warning(f"Could not initialize SHAP explainer. Details: {e}")
            return None, None

    return explainer, shap_values


def counseling_recommendations(risk_band: str) -> List[str]:
    """
    Counseling suggestions based on risk band.
    """
    if risk_band == "Low":
        return [
            "Maintain consistent weekly study reviews.",
            "Use calendar/time-blocking for routine tasks.",
            "Join peer study groups to stay engaged."
        ]
    elif risk_band == "Medium":
        return [
            "Connect with a mentor and set bi-weekly check-ins.",
            "Improve attendance with reminders and accountability.",
            "Track assignments via checklist; review progress weekly."
        ]
    elif risk_band == "High":
        return [
            "Schedule an immediate counselor meeting.",
            "Draft a personalized academic recovery plan.",
            "Arrange tutoring and frequent follow-ups (weekly)."
        ]
    else:
        return ["No recommendations available."]


# ============================
# Pages
# ============================

def sidebar_navigation() -> str:
    st.sidebar.markdown('<div class="sidebar-title">UmeedRise</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home",
            "📤 Upload + Train Model",
            "📊 Dashboard",
            "🧮 Predict New Students",
            "🔍 SHAP Explainability",
            "🧭 Counseling Plan",
        ],
        index=0,
        key="nav_radio"
    )

    st.sidebar.markdown("### ⚙️ Risk thresholds")
    low_thr = st.sidebar.slider("Low threshold", 0.0, 0.5, 0.33, 0.01, key="low_thr_slider")
    high_thr = st.sidebar.slider("High threshold", 0.5, 1.0, 0.66, 0.01, key="high_thr_slider")
    if high_thr <= low_thr:
        st.sidebar.warning("High threshold should be greater than Low threshold.")

    st.session_state["low_thr"] = low_thr
    st.session_state["high_thr"] = high_thr

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Confirm target detection before training.")
    return page



def home():
    st.markdown(f'<div class="app-title">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Predict dropout risk, explain drivers, and generate counseling plans — all in one modern dashboard.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Overview**")
        st.markdown("- Robust preprocessing: impute + scale + one-hot.")
        st.markdown("- Models: XGBoost (primary) with RandomForest fallback.")
        st.markdown("- Explainability via SHAP: summary + per-student.")
        st.markdown("- Plotly dashboard with risk and feature visuals.")
        st.markdown('<span class="chip">Auto-detection</span><span class="chip">XAI</span><span class="chip">Plotly</span><span class="chip">Counseling</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Workflow**")
        st.markdown("1. Upload dataset and confirm detected columns.")
        st.markdown("2. Train model, review metrics and confusion matrix.")
        st.markdown("3. Explore risk distributions and feature importance.")
        st.markdown("4. Predict new students and download CSV.")
        st.markdown("5. Explain predictions with SHAP.")
        st.markdown("6. Generate counseling plans automatically.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Quality & Safety**")
    st.markdown("- Graceful error handling and safe fallbacks.")
    st.markdown("- Modern gradient theme with rounded cards.")
    st.markdown("- Streamlit Cloud friendly; no deprecated sklearn APIs.")
    st.markdown('</div>', unsafe_allow_html=True)


def upload_and_train():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if file is None:
        st.info("Upload a CSV to proceed.")
        return

    df = load_data(file)
    if df is None or df.empty:
        st.error("Failed to load data or dataset is empty.")
        return

    # Light cleaning for common issues (e.g., stray spaces)
    df.columns = [c.strip() for c in df.columns]
    # Convert obvious numeric-looking columns to numeric (safe best-effort)
    for c in df.columns:
        if df[c].dtype == object:
            # Handle attendance with '%' or stray spaces
            if "%" in c or c.lower().strip() in ["attendance (in %)", "attendance"]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")

    st.session_state["df"] = df

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown('</div>', unsafe_allow_html=True)

    detected = detect_columns(df)
    target = detected["target"]
    student_id = detected["student_id"]
    numeric_cols = detected["numeric"]
    categorical_cols = detected["categorical"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔎 Column detection")

    colA, colB = st.columns(2)
    with colA:
        target = st.selectbox(
            "Target column",
            options=[None] + list(df.columns),
            index=(list(df.columns).index(target) + 1) if target in df.columns else 0
        )
    with colB:
        student_id = st.selectbox(
            "Student ID column (optional)",
            options=[None] + list(df.columns),
            index=(list(df.columns).index(student_id) + 1) if student_id in df.columns else 0
        )

    st.markdown("#### Feature selection")
    all_features = [c for c in df.columns if c != target and c != student_id]
    colC, colD = st.columns(2)
    with colC:
        numeric_cols = st.multiselect("Numeric features", options=all_features, default=[c for c in numeric_cols if c in all_features])
    with colD:
        categorical_cols = st.multiselect("Categorical features", options=all_features, default=[c for c in categorical_cols if c in all_features])
    st.caption("If you leave these empty, the app will auto-detect based on data types.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Missing values summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧹 Missing values summary")
    miss_df = pd.DataFrame({"column": df.columns, "missing_count": df.isna().sum().values})
    miss_df["missing_percent"] = (miss_df["missing_count"] / len(df) * 100).round(2)
    st.dataframe(miss_df.sort_values("missing_count", ascending=False), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Train
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Train model")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    run = st.button("Train")
    st.markdown('</div>', unsafe_allow_html=True)

    if not run:
        return

    if target is None:
        st.error("Target column is missing. Please select the target column.")
        return

    try:
        model, X_train_p, X_test_p, y_train, y_test, feature_names = train_model(
            df=df,
            target=target,
            student_id=student_id,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            test_size=test_size
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        return

    # Save to session
    st.session_state["model"] = model
    st.session_state["target"] = target
    st.session_state["student_id"] = student_id
    st.session_state["X_test_p"] = X_test_p
    st.session_state["y_test"] = y_test
    st.session_state["feature_names"] = feature_names

    metrics = evaluate(model, X_test_p, y_test)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance metrics")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    mc2.metric("Precision", f"{metrics['precision']:.3f}")
    mc3.metric("Recall", f"{metrics['recall']:.3f}")
    mc4.metric("F1", f"{metrics['f1']:.3f}")
    roc_display = metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else 0.0
    mc5.metric("ROC-AUC", f"{roc_display:.3f}")

    cm = np.array(metrics["confusion_matrix"])
    labels = [str(i) for i in sorted(np.unique(y_test))]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale="Purples",
        hovertemplate="True %{y}<br>Pred %{x}<br>Count %{z}<extra></extra>"
    ))
    cm_fig.update_layout(title="Confusion Matrix", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.success("Model trained successfully! Explore Dashboard, Predict, SHAP, and Counseling sections.")


def dashboard():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    X_test_p = st.session_state.get("X_test_p", None)
    y_test = st.session_state.get("y_test", None)
    feature_names = st.session_state.get("feature_names", [])
    df = st.session_state.get("df", None)
    student_id = st.session_state.get("student_id", None)
    feature_cols = st.session_state.get("feature_cols", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk distribution & overview")

    # Risk scores on test set
    try:
        y_proba = model.named_steps["clf"].predict_proba(X_test_p)
        risk_scores = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1)
    except Exception:
        preds = model.named_steps["clf"].predict(X_test_p)
        risk_scores = (preds == np.max(preds)).astype(float)

    low_thr = st.session_state.get("low_thr", 0.33)
    high_thr = st.session_state.get("high_thr", 0.66)
    risk_bands = [risk_from_proba(p, low_thr, high_thr) for p in risk_scores]

    hist_fig = px.histogram(
        x=risk_scores, nbins=30,
        color=risk_bands,
        color_discrete_map=RISK_COLORS,
        labels={"x": "Dropout risk score"},
        title="Risk score distribution"
    )
    hist_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist_fig, use_container_width=True)

    # Pie chart
    st.markdown("### 🥧 Risk categories")
    risk_counts = pd.Series(risk_bands).value_counts()
    pie_fig = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=RISK_COLORS,
        title="Risk category proportions"
    )
    pie_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Feature importance")
    importances = None
    try:
        importances = model.named_steps["clf"].feature_importances_
    except Exception:
        importances = None
        st.caption("Feature importances unavailable for this classifier. Try XGBoost/RandomForest.")

    if importances is not None and feature_names and len(importances) == len(feature_names):
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
        imp_fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top feature importances")
        imp_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(imp_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Top high-risk students
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧑‍🎓 Top high-risk students")
    if df is not None:
        try:
            preds_df = predict(model, df.copy(), feature_cols, low_thr, high_thr)
            cols_show = []
            if student_id and student_id in preds_df.columns:
                cols_show = [student_id, "dropout_risk_score", "dropout_risk_band"]
            else:
                preds_df["index"] = np.arange(len(preds_df))
                cols_show = ["index", "dropout_risk_score", "dropout_risk_band"]
            top_high = preds_df.sort_values("dropout_risk_score", ascending=False).head(20)[cols_show]
            st.dataframe(top_high, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute top high-risk table. Details: {e}")
    else:
        st.info("Upload and train to view high-risk students.")
    st.markdown('</div>', unsafe_allow_html=True)


def predict_new():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    feature_cols = st.session_state.get("feature_cols", [])
    low_thr = st.session_state.get("low_thr", 0.33)
    high_thr = st.session_state.get("high_thr", 0.66)
    student_id = st.session_state.get("student_id", None)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Predict for new students (CSV)")
    file = st.file_uploader("Upload CSV with new student records", type=["csv"], key="predict_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    if file is None:
        st.info("Upload a CSV of new students to get predictions.")
        return

    df_new = load_data(file)
    if df_new is None or df_new.empty:
        st.error("Failed to load prediction data or dataset is empty.")
        return

    df_new.columns = [c.strip() for c in df_new.columns]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Preview new data**")
    st.dataframe(df_new.head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        preds_df = predict(model, df_new.copy(), feature_cols, low_thr, high_thr)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Style risk column
    def risk_style(val):
        color = RISK_COLORS.get(val, "#333")
        return f"color: {color}; font-weight: 700;"

    styled = preds_df.style.applymap(lambda v: risk_style(v) if isinstance(v, str) and v in RISK_COLORS else "")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Predictions")
    st.dataframe(styled, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Download
    st.download_button(
        label="⬇️ Download predictions (CSV)",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="umeedrise_predictions.csv",
        mime="text/csv"
    )

    # Cache predictions for counseling page
    st.session_state["latest_predictions"] = preds_df


def shap_explainability():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not available. Please include 'shap' in requirements.")
        return

    model = st.session_state["model"]
    X_test_p = st.session_state.get("X_test_p", None)
    feature_names = st.session_state.get("feature_names", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 SHAP summary plot")
    try:
        explainer, shap_values = explain_with_shap(model, X_test_p, feature_names)
        if explainer is None or shap_values is None:
            st.info("SHAP explainer could not be initialized.")
        else:
            to_plot = shap_values
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                to_plot = shap_values[1]
            shap.summary_plot(to_plot, features=X_test_p, feature_names=feature_names, show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
    except Exception as e:
        st.warning(f"Failed to render SHAP summary plot. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Per-student explanation")
    try:
        if X_test_p is None or X_test_p.shape[0] == 0:
            st.info("No test instances available.")
        else:
            idx = st.number_input("Select test instance index", min_value=0, max_value=int(X_test_p.shape[0] - 1), value=0, step=1)
            explainer, shap_values = explain_with_shap(model, X_test_p, feature_names)
            if explainer is None or shap_values is None:
                st.info("SHAP explainer could not be initialized.")
            else:
                if isinstance(shap_values, list) and len(shap_values) >= 2:
                    values_for_instance = shap_values[1][idx]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    values_for_instance = shap_values[idx]
                    base_value = explainer.expected_value

                try:
                    shap.waterfall_plot(shap.Explanation(values=values_for_instance, base_values=base_value, data=X_test_p[idx], feature_names=feature_names), show=False)
                    st.pyplot(bbox_inches="tight", clear_figure=True)
                except Exception:
                    fig = shap.force_plot(base_value, values_for_instance, matplotlib=True)
                    st.pyplot(bbox_inches="tight", clear_figure=True)

                contrib_df = pd.DataFrame({"feature": feature_names, "shap_value": np.array(values_for_instance).flatten()}).sort_values("shap_value", ascending=False)
                st.markdown("#### Top contributing features")
                st.dataframe(contrib_df.head(10), use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to render per-student SHAP explanation. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def counseling_plan():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Counseling recommendation engine")
    st.caption("Use the Predict page to generate recommendations per student, or simulate below.")
    risk = st.selectbox("Select risk band", options=["Low", "Medium", "High"], index=2)
    recs = counseling_recommendations(risk)
    st.markdown(f"**Selected risk band:** <span class='risk-{risk.lower()}'>{risk}</span>", unsafe_allow_html=True)
    st.markdown("#### Suggested actions")
    for r in recs:
        st.markdown(f"- **Action:** {r}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Recommendations from latest predictions")
    preds_df = st.session_state.get("latest_predictions", None)
    if preds_df is not None and "dropout_risk_band" in preds_df.columns:
        out_df = preds_df.copy()
        out_df["recommendations"] = out_df["dropout_risk_band"].apply(lambda b: "; ".join(counseling_recommendations(b)))
        st.dataframe(out_df, use_container_width=True)
        st.download_button(
            label="⬇️ Download recommendations (CSV)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="umeedrise_recommendations.csv",
            mime="text/csv"
        )
    else:
        st.info("No cached predictions. Upload new student CSV on the Predict page first.")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================
# Entry Point
# ============================

def main():
    st.set_page_config(
        page_title="UmeedRise – Student Dropout Prediction",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_global_css()
    page = sidebar_navigation()

    if page == "🏠 Home":
        home()
    elif page == "📤 Upload + Train Model":
        upload_and_train()
    elif page == "📊 Dashboard":
        dashboard()
    elif page == "🧮 Predict New Students":
        predict_new()
    elif page == "🔍 SHAP Explainability":
        shap_explainability()
    elif page == "🧭 Counseling Plan":
        counseling_plan()
    else:
        home()

if __name__ == "__main__":
    main()
# UmeedRise – AI-Driven Student Dropout Prediction & Counseling System
# Streamlit app: production-ready, robust ML pipeline, SHAP explanations, Plotly dashboard, premium UI
# Author: GitHub Copilot
# Compatible with Streamlit Cloud

import io
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target

# Try optional libraries
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

APP_NAME = "UmeedRise – AI-Driven Student Dropout Prediction & Counseling System"

# ============================
# UI Styling
# ============================

PRIMARY_GRADIENT = """
background: linear-gradient(135deg, #6a73ff 0%, #8d7bff 40%, #b894ff 100%);
"""

CARD_STYLE = """
background-color: #ffffff;
border-radius: 18px;
box-shadow: 0 10px 30px rgba(0,0,0,0.10);
padding: 18px 20px;
border: 1px solid rgba(136, 136, 136, 0.15);
"""

CHIP_STYLE = """
display: inline-block;
padding: 5px 12px;
border-radius: 999px;
background: #f0f0ff;
color: #4c4cff;
font-weight: 600;
border: 1px solid rgba(120,120,255,0.3);
margin-right: 6px;
"""

RISK_COLORS = {
    "Low": "#34c759",      # green
    "Medium": "#ffcc00",   # yellow
    "High": "#ff3b30",     # red
}

def inject_global_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            {PRIMARY_GRADIENT}
            min-height: 100vh;
            background-attachment: fixed;
        }}
        section[data-testid="stSidebar"] > div {{
            {PRIMARY_GRADIENT}
            color: white !important;
        }}
        .sidebar-title {{
            font-weight: 800;
            font-size: 22px;
            margin-bottom: 8px;
            color: white;
        }}
        .app-title {{
            font-size: 34px;
            font-weight: 800;
            color: white;
            letter-spacing: 0.2px;
        }}
        .app-subtitle {{
            font-size: 16px;
            opacity: 0.95;
            color: white;
            font-weight: 500;
        }}
        .card {{
            {CARD_STYLE}
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(0,0,0,0.14);
        }}
        .chip {{
            {CHIP_STYLE}
        }}
        .metric-good {{ color: #34c759; font-weight: 700; }}
        .metric-mid {{ color: #ffcc00; font-weight: 700; }}
        .metric-bad {{ color: #ff3b30; font-weight: 700; }}
        .risk-low {{ color: #34c759; font-weight: 700; }}
        .risk-medium {{ color: #ffcc00; font-weight: 700; }}
        .risk-high {{ color: #ff3b30; font-weight: 700; }}
        .caption {{ font-size: 13px; opacity: 0.85; }}
        .small-note {{ font-size: 12px; opacity: 0.75; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ============================
# Core Functions
# ============================

def load_data(file: io.BytesIO) -> Optional[pd.DataFrame]:
    """
    Load CSV file robustly with encoding fallbacks.
    """
    if file is None:
        return None
    try:
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding=enc)
                return df
            except Exception:
                continue
        file.seek(0)
        df = pd.read_csv(file, engine="python")
        return df
    except Exception as e:
        st.warning(f"Could not read CSV. Please upload a valid CSV. Details: {e}")
        return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect target, student ID, numeric and categorical features.
    """
    if df is None or df.empty:
        return {"target": None, "student_id": None, "numeric": [], "categorical": []}

    cols = df.columns.tolist()
    cols_lower = [str(c).lower() for c in cols]

    # Target detection
    target_candidates = ["target", "dropout", "label", "risk", "outcome", "status", "y", "class"]
    target = None
    for tc in target_candidates:
        for i, c in enumerate(cols_lower):
            if tc == c or tc in c:
                target = cols[i]
                break
        if target:
            break

    # Student ID detection
    id_candidates = ["student_id", "id", "student", "roll", "rollno", "roll_no", "admission", "enrollment"]
    student_id = None
    for ic in id_candidates:
        for i, c in enumerate(cols_lower):
            if ic == c or ic in c:
                student_id = cols[i]
                break
        if student_id:
            break

    # Types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Exclude specials
    for special in [target, student_id]:
        if special in numeric_cols:
            numeric_cols.remove(special)
        if special in categorical_cols:
            categorical_cols.remove(special)

    # Heuristic fallback: if no target, look for binary-like columns
    if target is None:
        for col in cols:
            unique_vals = pd.Series(df[col]).dropna().unique()
            if len(unique_vals) == 2 and col != student_id:
                target = col
                break

    return {"target": target, "student_id": student_id, "numeric": numeric_cols, "categorical": categorical_cols}


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessing:
    - Numeric: median impute + StandardScaler
    - Categorical: constant impute + OneHotEncoder (sparse_output=False)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def train_model(
    df: pd.DataFrame,
    target: str,
    student_id: Optional[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train classifier with XGBoost primary and RandomForest fallback.
    Returns pipeline, processed X_train/X_test, y_train/y_test, and transformed feature names.
    """
    if target is None or target not in df.columns:
        raise ValueError("Target column missing or not found.")

    # Feature columns
    feature_cols = [c for c in df.columns if c != target and c != student_id]
    if not numeric_cols and not categorical_cols:
        # Auto split based on dtypes on the chosen features
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X = df[feature_cols].copy()
    y_raw = df[target].copy()

    # Encode target
    y_type = type_of_target(y_raw)
    if y_type in ["binary", "multiclass"]:
        if not np.issubdtype(y_raw.dtype, np.number):
            y, _ = pd.factorize(y_raw)
        else:
            y = y_raw.astype(int)
    elif y_type == "continuous":
        st.warning("Detected continuous target; converting to binary using median threshold.")
        thresh = pd.to_numeric(y_raw, errors="coerce").median()
        y = (pd.to_numeric(y_raw, errors="coerce") >= thresh).astype(int)
    else:
        y = pd.factorize(y_raw)[0]

    # Stratified split if possible
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Classifier choice
    if XGB_AVAILABLE:
        objective = "binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob"
        classifier = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective=objective,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", classifier)])
    model.fit(X_train, y_train)

    # Feature names post-transform
    feature_names = []
    try:
        preprocessor.fit(X_train)
        num_features = numeric_cols
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
        except Exception:
            cat_features = categorical_cols
        feature_names = num_features + cat_features
    except Exception:
        try:
            n_out = model.named_steps["preprocessor"].transform(X_train).shape[1]
            feature_names = [f"f_{i}" for i in range(n_out)]
        except Exception:
            feature_names = []

    X_train_processed = model.named_steps["preprocessor"].transform(X_train)
    X_test_processed = model.named_steps["preprocessor"].transform(X_test)

    # Cache helpful info
    st.session_state["feature_cols"] = feature_cols
    st.session_state["numeric_cols"] = numeric_cols
    st.session_state["categorical_cols"] = categorical_cols

    return model, X_train_processed, X_test_processed, y_train, y_test, feature_names


def evaluate(model: Pipeline, X_test_processed: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance with robust handling for binary/multiclass.
    """
    y_pred = model.named_steps["clf"].predict(X_test_processed)
    y_proba = None
    try:
        y_proba = model.named_steps["clf"].predict_proba(X_test_processed)
    except Exception:
        y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred))
    }
    average_strategy = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    metrics["precision"] = float(precision_score(y_test, y_pred, average=average_strategy, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, average=average_strategy, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, average=average_strategy, zero_division=0))

    try:
        if y_proba is not None:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            metrics["roc_auc"] = float(auc)
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def risk_from_proba(p: float, low_thr: float = 0.33, high_thr: float = 0.66) -> str:
    """
    Map probability to Low/Medium/High risk.
    """
    if p < low_thr:
        return "Low"
    elif p < high_thr:
        return "Medium"
    else:
        return "High"


def predict(
    model: Pipeline,
    df_input: pd.DataFrame,
    feature_cols: List[str],
    low_thr: float = 0.33,
    high_thr: float = 0.66
) -> pd.DataFrame:
    """
    Predict dropout risk for new data with colored risk bands and scores.
    """
    # Align columns
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = np.nan

    X_new = df_input[feature_cols].copy()

    try:
        proba = model.predict_proba(X_new)
        if proba.shape[1] == 2:
            risk_p = proba[:, 1]
        else:
            risk_p = proba.max(axis=1)
    except Exception:
        preds = model.predict(X_new)
        # Fallback: treat positive class as risk
        risk_p = (preds == np.max(preds)).astype(float)

    y_pred = model.predict(X_new)
    risks = [risk_from_proba(p, low_thr, high_thr) for p in risk_p]

    result = df_input.copy()
    result["predicted_class"] = y_pred
    result["dropout_risk_score"] = np.round(risk_p, 4)
    result["dropout_risk_band"] = risks

    return result


def explain_with_shap(model: Pipeline, X_processed: np.ndarray, feature_names: List[str]):
    """
    Initialize SHAP explainer and values. TreeExplainer preferred.
    """
    if not SHAP_AVAILABLE:
        st.warning("SHAP not available. Include 'shap' in requirements.")
        return None, None

    clf = model.named_steps["clf"]
    explainer = None
    shap_values = None

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_processed)
    except Exception:
        try:
            explainer = shap.Explainer(clf.predict, model.named_steps["preprocessor"].transform)
            shap_values = explainer(X_processed)
        except Exception as e:
            st.warning(f"Could not initialize SHAP explainer. Details: {e}")
            return None, None

    return explainer, shap_values


def counseling_recommendations(risk_band: str) -> List[str]:
    """
    Counseling suggestions based on risk band.
    """
    if risk_band == "Low":
        return [
            "Maintain consistent weekly study reviews.",
            "Use calendar/time-blocking for routine tasks.",
            "Join peer study groups to stay engaged."
        ]
    elif risk_band == "Medium":
        return [
            "Connect with a mentor and set bi-weekly check-ins.",
            "Improve attendance with reminders and accountability.",
            "Track assignments via checklist; review progress weekly."
        ]
    elif risk_band == "High":
        return [
            "Schedule an immediate counselor meeting.",
            "Draft a personalized academic recovery plan.",
            "Arrange tutoring and frequent follow-ups (weekly)."
        ]
    else:
        return ["No recommendations available."]


# ============================
# Pages
# ============================

def sidebar_navigation() -> str:
    st.sidebar.markdown('<div class="sidebar-title">UmeedRise</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home",
            "📤 Upload + Train Model",
            "📊 Dashboard",
            "🧮 Predict New Students",
            "🔍 SHAP Explainability",
            "🧭 Counseling Plan",
        ],
        index=0
    )

    st.sidebar.markdown("### ⚙️ Risk thresholds")
    low_thr = st.sidebar.slider("Low threshold", 0.0, 0.5, 0.33, 0.01)
    high_thr = st.sidebar.slider("High threshold", 0.5, 1.0, 0.66, 0.01)
    if high_thr <= low_thr:
        st.sidebar.warning("High threshold should be greater than Low threshold.")

    st.session_state["low_thr"] = low_thr
    st.session_state["high_thr"] = high_thr

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Confirm target detection before training.")
    return page


def home():
    st.markdown(f'<div class="app-title">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Predict dropout risk, explain drivers, and generate counseling plans — all in one modern dashboard.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Overview**")
        st.markdown("- Robust preprocessing: impute + scale + one-hot.")
        st.markdown("- Models: XGBoost (primary) with RandomForest fallback.")
        st.markdown("- Explainability via SHAP: summary + per-student.")
        st.markdown("- Plotly dashboard with risk and feature visuals.")
        st.markdown('<span class="chip">Auto-detection</span><span class="chip">XAI</span><span class="chip">Plotly</span><span class="chip">Counseling</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Workflow**")
        st.markdown("1. Upload dataset and confirm detected columns.")
        st.markdown("2. Train model, review metrics and confusion matrix.")
        st.markdown("3. Explore risk distributions and feature importance.")
        st.markdown("4. Predict new students and download CSV.")
        st.markdown("5. Explain predictions with SHAP.")
        st.markdown("6. Generate counseling plans automatically.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Quality & Safety**")
    st.markdown("- Graceful error handling and safe fallbacks.")
    st.markdown("- Modern gradient theme with rounded cards.")
    st.markdown("- Streamlit Cloud friendly; no deprecated sklearn APIs.")
    st.markdown('</div>', unsafe_allow_html=True)


def upload_and_train():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if file is None:
        st.info("Upload a CSV to proceed.")
        return

    df = load_data(file)
    if df is None or df.empty:
        st.error("Failed to load data or dataset is empty.")
        return

    # Light cleaning for common issues (e.g., stray spaces)
    df.columns = [c.strip() for c in df.columns]
    # Convert obvious numeric-looking columns to numeric (safe best-effort)
    for c in df.columns:
        if df[c].dtype == object:
            # Handle attendance with '%' or stray spaces
            if "%" in c or c.lower().strip() in ["attendance (in %)", "attendance"]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")

    st.session_state["df"] = df

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown('</div>', unsafe_allow_html=True)

    detected = detect_columns(df)
    target = detected["target"]
    student_id = detected["student_id"]
    numeric_cols = detected["numeric"]
    categorical_cols = detected["categorical"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔎 Column detection")

    colA, colB = st.columns(2)
    with colA:
        target = st.selectbox(
            "Target column",
            options=[None] + list(df.columns),
            index=(list(df.columns).index(target) + 1) if target in df.columns else 0
        )
    with colB:
        student_id = st.selectbox(
            "Student ID column (optional)",
            options=[None] + list(df.columns),
            index=(list(df.columns).index(student_id) + 1) if student_id in df.columns else 0
        )

    st.markdown("#### Feature selection")
    all_features = [c for c in df.columns if c != target and c != student_id]
    colC, colD = st.columns(2)
    with colC:
        numeric_cols = st.multiselect("Numeric features", options=all_features, default=[c for c in numeric_cols if c in all_features])
    with colD:
        categorical_cols = st.multiselect("Categorical features", options=all_features, default=[c for c in categorical_cols if c in all_features])
    st.caption("If you leave these empty, the app will auto-detect based on data types.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Missing values summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧹 Missing values summary")
    miss_df = pd.DataFrame({"column": df.columns, "missing_count": df.isna().sum().values})
    miss_df["missing_percent"] = (miss_df["missing_count"] / len(df) * 100).round(2)
    st.dataframe(miss_df.sort_values("missing_count", ascending=False), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Train
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Train model")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    run = st.button("Train")
    st.markdown('</div>', unsafe_allow_html=True)

    if not run:
        return

    if target is None:
        st.error("Target column is missing. Please select the target column.")
        return

    try:
        model, X_train_p, X_test_p, y_train, y_test, feature_names = train_model(
            df=df,
            target=target,
            student_id=student_id,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            test_size=test_size
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        return

    # Save to session
    st.session_state["model"] = model
    st.session_state["target"] = target
    st.session_state["student_id"] = student_id
    st.session_state["X_test_p"] = X_test_p
    st.session_state["y_test"] = y_test
    st.session_state["feature_names"] = feature_names

    metrics = evaluate(model, X_test_p, y_test)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance metrics")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    mc2.metric("Precision", f"{metrics['precision']:.3f}")
    mc3.metric("Recall", f"{metrics['recall']:.3f}")
    mc4.metric("F1", f"{metrics['f1']:.3f}")
    roc_display = metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else 0.0
    mc5.metric("ROC-AUC", f"{roc_display:.3f}")

    cm = np.array(metrics["confusion_matrix"])
    labels = [str(i) for i in sorted(np.unique(y_test))]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale="Purples",
        hovertemplate="True %{y}<br>Pred %{x}<br>Count %{z}<extra></extra>"
    ))
    cm_fig.update_layout(title="Confusion Matrix", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.success("Model trained successfully! Explore Dashboard, Predict, SHAP, and Counseling sections.")


def dashboard():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    X_test_p = st.session_state.get("X_test_p", None)
    y_test = st.session_state.get("y_test", None)
    feature_names = st.session_state.get("feature_names", [])
    df = st.session_state.get("df", None)
    student_id = st.session_state.get("student_id", None)
    feature_cols = st.session_state.get("feature_cols", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk distribution & overview")

    # Risk scores on test set
    try:
        y_proba = model.named_steps["clf"].predict_proba(X_test_p)
        risk_scores = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1)
    except Exception:
        preds = model.named_steps["clf"].predict(X_test_p)
        risk_scores = (preds == np.max(preds)).astype(float)

    low_thr = st.session_state.get("low_thr", 0.33)
    high_thr = st.session_state.get("high_thr", 0.66)
    risk_bands = [risk_from_proba(p, low_thr, high_thr) for p in risk_scores]

    hist_fig = px.histogram(
        x=risk_scores, nbins=30,
        color=risk_bands,
        color_discrete_map=RISK_COLORS,
        labels={"x": "Dropout risk score"},
        title="Risk score distribution"
    )
    hist_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist_fig, use_container_width=True)

    # Pie chart
    st.markdown("### 🥧 Risk categories")
    risk_counts = pd.Series(risk_bands).value_counts()
    pie_fig = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=RISK_COLORS,
        title="Risk category proportions"
    )
    pie_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Feature importance")
    importances = None
    try:
        importances = model.named_steps["clf"].feature_importances_
    except Exception:
        importances = None
        st.caption("Feature importances unavailable for this classifier. Try XGBoost/RandomForest.")

    if importances is not None and feature_names and len(importances) == len(feature_names):
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
        imp_fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top feature importances")
        imp_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(imp_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Top high-risk students
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧑‍🎓 Top high-risk students")
    if df is not None:
        try:
            preds_df = predict(model, df.copy(), feature_cols, low_thr, high_thr)
            cols_show = []
            if student_id and student_id in preds_df.columns:
                cols_show = [student_id, "dropout_risk_score", "dropout_risk_band"]
            else:
                preds_df["index"] = np.arange(len(preds_df))
                cols_show = ["index", "dropout_risk_score", "dropout_risk_band"]
            top_high = preds_df.sort_values("dropout_risk_score", ascending=False).head(20)[cols_show]
            st.dataframe(top_high, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute top high-risk table. Details: {e}")
    else:
        st.info("Upload and train to view high-risk students.")
    st.markdown('</div>', unsafe_allow_html=True)


def predict_new():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    model = st.session_state["model"]
    feature_cols = st.session_state.get("feature_cols", [])
    low_thr = st.session_state.get("low_thr", 0.33)
    high_thr = st.session_state.get("high_thr", 0.66)
    student_id = st.session_state.get("student_id", None)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Predict for new students (CSV)")
    file = st.file_uploader("Upload CSV with new student records", type=["csv"], key="predict_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    if file is None:
        st.info("Upload a CSV of new students to get predictions.")
        return

    df_new = load_data(file)
    if df_new is None or df_new.empty:
        st.error("Failed to load prediction data or dataset is empty.")
        return

    df_new.columns = [c.strip() for c in df_new.columns]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Preview new data**")
    st.dataframe(df_new.head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        preds_df = predict(model, df_new.copy(), feature_cols, low_thr, high_thr)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Style risk column
    def risk_style(val):
        color = RISK_COLORS.get(val, "#333")
        return f"color: {color}; font-weight: 700;"

    styled = preds_df.style.applymap(lambda v: risk_style(v) if isinstance(v, str) and v in RISK_COLORS else "")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Predictions")
    st.dataframe(styled, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Download
    st.download_button(
        label="⬇️ Download predictions (CSV)",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="umeedrise_predictions.csv",
        mime="text/csv"
    )

    # Cache predictions for counseling page
    st.session_state["latest_predictions"] = preds_df


def shap_explainability():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not available. Please include 'shap' in requirements.")
        return

    model = st.session_state["model"]
    X_test_p = st.session_state.get("X_test_p", None)
    feature_names = st.session_state.get("feature_names", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 SHAP summary plot")
    try:
        explainer, shap_values = explain_with_shap(model, X_test_p, feature_names)
        if explainer is None or shap_values is None:
            st.info("SHAP explainer could not be initialized.")
        else:
            to_plot = shap_values
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                to_plot = shap_values[1]
            shap.summary_plot(to_plot, features=X_test_p, feature_names=feature_names, show=False)
            st.pyplot(bbox_inches="tight", clear_figure=True)
    except Exception as e:
        st.warning(f"Failed to render SHAP summary plot. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Per-student explanation")
    try:
        if X_test_p is None or X_test_p.shape[0] == 0:
            st.info("No test instances available.")
        else:
            idx = st.number_input("Select test instance index", min_value=0, max_value=int(X_test_p.shape[0] - 1), value=0, step=1)
            explainer, shap_values = explain_with_shap(model, X_test_p, feature_names)
            if explainer is None or shap_values is None:
                st.info("SHAP explainer could not be initialized.")
            else:
                if isinstance(shap_values, list) and len(shap_values) >= 2:
                    values_for_instance = shap_values[1][idx]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    values_for_instance = shap_values[idx]
                    base_value = explainer.expected_value

                try:
                    shap.waterfall_plot(shap.Explanation(values=values_for_instance, base_values=base_value, data=X_test_p[idx], feature_names=feature_names), show=False)
                    st.pyplot(bbox_inches="tight", clear_figure=True)
                except Exception:
                    fig = shap.force_plot(base_value, values_for_instance, matplotlib=True)
                    st.pyplot(bbox_inches="tight", clear_figure=True)

                contrib_df = pd.DataFrame({"feature": feature_names, "shap_value": np.array(values_for_instance).flatten()}).sort_values("shap_value", ascending=False)
                st.markdown("#### Top contributing features")
                st.dataframe(contrib_df.head(10), use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to render per-student SHAP explanation. Details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def counseling_plan():
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("Model is not trained yet. Please upload a dataset and train the model first.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Counseling recommendation engine")
    st.caption("Use the Predict page to generate recommendations per student, or simulate below.")
    risk = st.selectbox("Select risk band", options=["Low", "Medium", "High"], index=2)
    recs = counseling_recommendations(risk)
    st.markdown(f"**Selected risk band:** <span class='risk-{risk.lower()}'>{risk}</span>", unsafe_allow_html=True)
    st.markdown("#### Suggested actions")
    for r in recs:
        st.markdown(f"- **Action:** {r}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Recommendations from latest predictions")
    preds_df = st.session_state.get("latest_predictions", None)
    if preds_df is not None and "dropout_risk_band" in preds_df.columns:
        out_df = preds_df.copy()
        out_df["recommendations"] = out_df["dropout_risk_band"].apply(lambda b: "; ".join(counseling_recommendations(b)))
        st.dataframe(out_df, use_container_width=True)
        st.download_button(
            label="⬇️ Download recommendations (CSV)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="umeedrise_recommendations.csv",
            mime="text/csv"
        )
    else:
        st.info("No cached predictions. Upload new student CSV on the Predict page first.")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================
# Entry Point
# ============================

def main():
    st.set_page_config(
        page_title="UmeedRise – Student Dropout Prediction",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_global_css()
    page = sidebar_navigation()

    if page == "🏠 Home":
        home()
    elif page == "📤 Upload + Train Model":
        upload_and_train()
    elif page == "📊 Dashboard":
        dashboard()
    elif page == "🧮 Predict New Students":
        predict_new()
    elif page == "🔍 SHAP Explainability":
        shap_explainability()
    elif page == "🧭 Counseling Plan":
        counseling_plan()
    else:
        home()

if __name__ == "__main__":
    main()
