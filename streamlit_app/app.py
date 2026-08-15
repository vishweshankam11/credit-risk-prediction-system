import os
import json
import io
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import database as db

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# REFERENCE-IMAGE STYLE
# ============================================================
st.markdown("""
<style>
.stApp{background:#070d16;color:#f8fafc}
.main{background:#070d16}
.block-container{max-width:1400px;padding:4.5rem 2.2rem 2.5rem}
#MainMenu,footer{visibility:hidden}
hr{border-color:#334155!important}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#172235 0%,#101827 100%);
    border-right:1px solid #334155;min-width:250px
}
section[data-testid="stSidebar"]>div{background:transparent}
section[data-testid="stSidebar"] p{color:#94a3b8}
section[data-testid="stSidebar"] hr{border-color:#334155!important}

.sidebar-brand{text-align:center;padding:8px 0 18px}
.sidebar-shield{font-size:3.4rem;line-height:1;margin-bottom:14px}
.sidebar-title{color:#f8fafc;font-size:1.35rem;font-weight:800}
.sidebar-subtitle{color:#7182a0;font-size:1rem;margin-top:8px}

.user-card{
    background:linear-gradient(135deg,#f8fbff,#dcecff);
    border:1px solid #60a5fa;border-radius:15px;padding:18px;
    margin:12px 0;box-shadow:0 8px 25px rgba(0,0,0,.28)
}
.user-icon{color:#2563eb;font-size:1.8rem;line-height:1}
.user-title{color:#1471d9!important;font-size:.95rem;font-weight:800;margin:7px 0}
.username{
    display:inline-block;background:#111827;color:#4ade80;
    padding:5px 9px;border-radius:5px;font-family:monospace;font-weight:700
}

section[data-testid="stSidebar"] .stButton>button{
    background:rgba(20,30,47,.75);color:#f8fafc;border:1px solid #334155;
    border-radius:10px;min-height:48px;font-size:.95rem;font-weight:600;
    transition:.2s;margin-bottom:3px
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:#1d65d8;border-color:#3b82f6;color:#fff;transform:translateY(-1px)
}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#1675ee,#1765d1);
    border-color:#3987ff;color:#fff
}

.main-header{padding:.1rem 0 1rem}
.main-title{font-size:2.35rem;font-weight:800;color:#f8fafc;line-height:1.1}
.main-subtitle{font-size:1rem;color:#94a3b8;margin-top:12px}

.welcome-banner{
    background:linear-gradient(90deg,#092c22,#0c3c2c);
    border:1px solid #198b4b;border-radius:11px;padding:17px 21px;
    color:#4ade80;font-size:1rem;font-weight:700;margin:4px 0 18px
}

.metric-card{
    min-height:132px;border-radius:14px;padding:19px 20px;border:1px solid;
    box-shadow:0 8px 25px rgba(0,0,0,.22);transition:.2s
}
.metric-card:hover{transform:translateY(-3px);box-shadow:0 13px 32px rgba(0,0,0,.32)}
.metric-blue{background:linear-gradient(135deg,#f5faff,#dcecff);border-color:#63a5fa}
.metric-red{background:linear-gradient(135deg,#fff7f7,#ffe2e5);border-color:#fb7185}
.metric-green{background:linear-gradient(135deg,#f7fff9,#ddf8e4);border-color:#66d88c}
.metric-purple{background:linear-gradient(135deg,#fbf9ff,#eee5ff);border-color:#b58ae8}
.metric-title{font-size:.94rem;font-weight:800;margin-bottom:17px}
.metric-blue .metric-title{color:#1774dc}
.metric-red .metric-title{color:#dc2626}
.metric-green .metric-title{color:#15803d}
.metric-purple .metric-title{color:#7044b8}
.metric-value{font-size:2rem;font-weight:850;line-height:1;color:#111827}

.content-card{
    background:linear-gradient(145deg,#111a28,#0d1521);
    border:1px solid #2e3c50;border-radius:12px;padding:17px 19px;
    box-shadow:0 8px 25px rgba(0,0,0,.23);margin-top:18px
}
.card-title{color:#f8fafc;font-size:1.2rem;font-weight:800}
.section-title{font-size:1.35rem;font-weight:800;color:#f8fafc;margin:1rem 0 .8rem}
.section-description{color:#94a3b8;margin-bottom:1.2rem}

.prediction-table{
    width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;
    border:1px solid #334155;border-radius:10px;background:#0e1724;color:#f8fafc
}
.prediction-table th{
    background:#1d2a3d;color:#f8fafc;padding:13px 14px;text-align:left;
    font-weight:800;border-bottom:1px solid #475569
}
.prediction-table td{padding:12px 14px;border-bottom:1px solid #273548;color:#e2e8f0}
.prediction-table tr:last-child td{border-bottom:none}
.prediction-table tr:hover td{background:#152237}
.risk-low-badge,.risk-high-badge{
    display:inline-block;padding:5px 10px;border-radius:7px;font-weight:800;white-space:nowrap
}
.risk-low-badge{background:#064e3b;color:#4ade80}
.risk-high-badge{background:#651b23;color:#ff7d88}
.probability-low{color:#4ade80;font-weight:800}
.probability-high{color:#ff6b76;font-weight:800}

.risk-high{
    background:linear-gradient(135deg,#450a0a,#7f1d1d);
    border:1px solid #ef4444;border-radius:16px;padding:1.5rem;text-align:center;color:white
}
.risk-low{
    background:linear-gradient(135deg,#052e16,#14532d);
    border:1px solid #22c55e;border-radius:16px;padding:1.5rem;text-align:center;color:white
}
.risk-percentage{font-size:2.8rem;font-weight:800;margin:.5rem 0}
.risk-label{font-size:1.15rem;font-weight:800}

.info-card{
    background:linear-gradient(145deg,#111a28,#0d1521);
    border:1px solid #334155;border-radius:14px;padding:1.2rem;
    margin-bottom:1rem;color:#e2e8f0;box-shadow:0 5px 20px rgba(0,0,0,.2)
}
.info-card b,.info-card h3{color:#f8fafc}

input,textarea{background:#111827!important;color:#f8fafc!important;border-color:#475569!important}
div[data-baseweb="select"]>div{background:#111827!important;color:#f8fafc!important;border-color:#475569!important}
label{color:#dbeafe!important}
div[data-testid="stDataFrame"]{border:1px solid #334155;border-radius:10px;overflow:hidden}

.footer{text-align:center;color:#64748b;font-size:.8rem;padding:1.5rem 0}
.login-logo{text-align:center;margin-top:40px;margin-bottom:20px}
.login-logo-icon{font-size:4rem}
.login-title{color:#f8fafc;font-size:2rem;font-weight:800}
.login-subtitle{color:#94a3b8;font-size:1rem}
</style>
""", unsafe_allow_html=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

DB_CONFIG = {
    "host": st.secrets["mysql"]["host"],
    "port": int(st.secrets["mysql"].get("port", 3306)),
    "user": st.secrets["mysql"]["user"],
    "password": st.secrets["mysql"]["password"],
    "database": st.secrets["mysql"]["database"],
}

DEFAULT_SESSION = {
    "logged_in": False,
    "username": None,
    "page": "Dashboard",
    "prediction_done": False,
    "latest_probability": None,
    "latest_prediction": None,
    "latest_input_df": None,
    "latest_input_dict": None,
    "latest_shap_values": None,
}
for key,value in DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# DATABASE / CACHE
# ============================================================
@st.cache_resource
def initialize_database():
    db.init_db(DB_CONFIG)
    return True

@st.cache_data(ttl=20, show_spinner=False)
def load_dashboard_summary():
    """Load only aggregate dashboard statistics from MySQL."""
    return db.get_prediction_summary(DB_CONFIG)

@st.cache_data(ttl=20, show_spinner=False)
def load_recent_predictions(limit=8):
    """Load only the rows required for the dashboard."""
    return pd.DataFrame(db.get_recent_predictions(DB_CONFIG, limit=limit))

@st.cache_data(ttl=20, show_spinner=False)
def load_analytics_recent(limit=100):
    """
    Analytics uses a small recent window for charts.
    This avoids loading the complete history when opening the page.
    """
    return pd.DataFrame(db.get_recent_predictions(DB_CONFIG, limit=limit))

@st.cache_data(ttl=20, show_spinner=False)
def load_prediction_history():
    """
    Full history is intentionally loaded only after the user asks for it
    from the Analytics page.
    """
    return pd.DataFrame(db.get_all_predictions(DB_CONFIG))

def clear_database_caches():
    load_dashboard_summary.clear()
    load_recent_predictions.clear()
    load_analytics_recent.clear()
    load_prediction_history.clear()

# ============================================================
# HELPERS
# ============================================================
def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError,ValueError):
        return default

def prepare_recent_dataframe(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "prediction_timestamp","predicted_by",
            "risk_prediction","risk_probability"
        ])
    df = df.copy()
    aliases = {
        "timestamp":"prediction_timestamp",
        "created_at":"prediction_timestamp",
        "predicted_at":"prediction_timestamp",
        "username":"predicted_by",
        "user":"predicted_by",
        "prediction":"risk_prediction",
        "risk":"risk_prediction",
        "probability":"risk_probability",
        "default_probability":"risk_probability",
    }
    for old,new in aliases.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    defaults_map = {
        "prediction_timestamp":"",
        "predicted_by":"",
        "risk_prediction":0,
        "risk_probability":0.0,
    }
    for col,default in defaults_map.items():
        if col not in df.columns:
            df[col] = default
    df["risk_prediction"] = pd.to_numeric(
        df["risk_prediction"],errors="coerce"
    ).fillna(0).astype(int)
    df["risk_probability"] = pd.to_numeric(
        df["risk_probability"],errors="coerce"
    ).fillna(0.0)
    return df

def render_recent_predictions(df):
    df = prepare_recent_dataframe(df)
    if df.empty:
        st.info("No recent predictions available.")
        return
    rows = []
    for _,row in df.head(8).iterrows():
        prediction = int(row["risk_prediction"])
        probability = safe_float(row["risk_probability"])
        probability_display = probability*100 if probability <= 1 else probability
        risk_html = (
            '<span class="risk-high-badge">🔴 High Risk</span>'
            if prediction == 1 else
            '<span class="risk-low-badge">🟢 Low Risk</span>'
        )
        pclass = "probability-high" if prediction == 1 else "probability-low"
        timestamp = row["prediction_timestamp"]
        try:
            timestamp = pd.to_datetime(timestamp).strftime("%d-%m-%Y %H:%M:%S")
        except Exception:
            timestamp = str(timestamp)
        rows.append(
            f"<tr><td>{timestamp}</td><td>{row['predicted_by']}</td>"
            f"<td>{risk_html}</td><td class='{pclass}'>{probability_display:.1f}%</td></tr>"
        )
    st.markdown(
        f"""<table class="prediction-table">
        <thead><tr><th>Timestamp</th><th>Predicted By</th><th>Risk</th><th>Probability</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table>""",
        unsafe_allow_html=True,
    )

def make_risk_donut(low_risk,high_risk):
    fig,ax = plt.subplots(figsize=(5.2,4.2))
    counts=[low_risk,high_risk]
    if sum(counts)>0:
        _,_,texts=ax.pie(
            counts,colors=["#22c55e","#ef4444"],startangle=90,
            counterclock=False,autopct=lambda p:f"{p:.0f}%" if p>0 else "",
            pctdistance=.78,wedgeprops=dict(width=.48,edgecolor="#0f1724",linewidth=1.5)
        )
        for text in texts:
            text.set_color("white");text.set_fontsize(12);text.set_fontweight("bold")
        ax.add_artist(plt.Circle((0,0),.48,fc="#111a28"))
    ax.set_aspect("equal");ax.set_facecolor("#111a28");fig.patch.set_facecolor("#111a28")
    plt.tight_layout()
    return fig

def make_probability_histogram(df):
    df=prepare_recent_dataframe(df)
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    if not df.empty:
        values=df["risk_probability"].to_numpy(dtype=float)
        values=np.where(values<=1,values*100,values)
        ax.hist(values,bins=10,color="#2f80ed",edgecolor="#0f1724")
    ax.set_xlabel("Default Probability (%)",color="#f8fafc")
    ax.set_ylabel("Number of Predictions",color="#f8fafc")
    ax.tick_params(colors="#e2e8f0")
    ax.set_facecolor("#111a28");fig.patch.set_facecolor("#111a28")
    for spine in ax.spines.values(): spine.set_color("#475569")
    ax.grid(axis="y",color="#334155",linestyle="--",alpha=.7)
    plt.tight_layout()
    return fig

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_artifacts():
    """
    Load model and JSON artifacts only.
    SHAP is loaded lazily when a prediction is actually requested.
    """
    model=xgb.XGBClassifier()
    model.load_model(os.path.join(APP_DIR,"model.json"))

    with open(os.path.join(APP_DIR,"feature_columns.json"),encoding="utf-8") as f:
        feature_columns=json.load(f)
    with open(os.path.join(APP_DIR,"defaults.json"),encoding="utf-8") as f:
        defaults=json.load(f)
    with open(os.path.join(APP_DIR,"dataset_averages.json"),encoding="utf-8") as f:
        dataset_averages=json.load(f)

    return model,feature_columns,defaults,dataset_averages

@st.cache_resource
def load_shap_explainer():
    """Create the SHAP TreeExplainer once and reuse it."""
    model,_,_,_=load_artifacts()
    return shap.TreeExplainer(model)

@st.cache_data(show_spinner=False)
def build_shap_waterfall_png(shap_values,input_values,feature_columns,base_value):
    """
    Create the waterfall image once during prediction.
    Explainability can then display the stored PNG immediately.
    """
    values=np.asarray(shap_values,dtype=float).reshape(-1)
    data=np.asarray(input_values).reshape(-1)

    explanation=shap.Explanation(
        values=values,
        base_values=float(base_value),
        data=data,
        feature_names=list(feature_columns)
    )

    shap.plots.waterfall(
        explanation,
        show=False,
        max_display=10
    )

    fig=plt.gcf()
    buffer=io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# LOGIN
# ============================================================
if not st.session_state.logged_in:
    st.markdown("""
    <div class="login-logo">
        <div class="login-logo-icon">🛡️</div>
        <div class="login-title">Credit Risk Prediction System</div>
        <div class="login-subtitle">Secure access for authorized users</div>
    </div>""",unsafe_allow_html=True)

    _,center,_=st.columns([1,1.2,1])
    with center:
        with st.form("login_form"):
            username=st.text_input("Username",placeholder="Enter your username")
            password=st.text_input("Password",type="password",placeholder="Enter your password")
            submitted=st.form_submit_button("🔐 Sign In",type="primary",use_container_width=True)
        if submitted:
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                try:
                    with st.spinner("Authenticating..."):
                        valid_user=db.verify_admin(DB_CONFIG,username.strip(),password)
                    if valid_user:
                        st.session_state.logged_in=True
                        st.session_state.username=username.strip()
                        st.session_state.page="Dashboard"
                        try: initialize_database()
                        except Exception: pass
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception:
                    st.error("Unable to connect to the authentication service.")
    st.markdown("""
    <div class="footer">🔒 Authorized Access Only &nbsp;•&nbsp; Credit Risk Assessment Platform</div>
    """,unsafe_allow_html=True)
    st.stop()

try:
    model,feature_columns,defaults,dataset_averages=load_artifacts()
except Exception:
    st.error(
        "Model files could not be loaded. Make sure model.json, "
        "feature_columns.json, defaults.json and dataset_averages.json are present."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-shield">🛡️</div>
        <div class="sidebar-title">Credit Risk</div>
        <div class="sidebar-subtitle">Prediction System</div>
    </div>""",unsafe_allow_html=True)
    st.divider()
    st.markdown(
        f"""<div class="user-card">
        <div class="user-icon">👤</div>
        <div class="user-title">Logged in as</div>
        <div class="username">{st.session_state.username}</div>
        </div>""",unsafe_allow_html=True)
    st.divider()

    pages={
        "🏠 Dashboard":"Dashboard",
        "🔍 Risk Prediction":"Risk Prediction",
        "🧠 Explainability":"Explainability",
        "📊 Analytics":"Analytics",
        "🤖 Model Information":"Model Information",
        "ℹ️ About":"About",
    }
    for label,page_name in pages.items():
        if st.button(
            label,use_container_width=True,key=f"nav_{page_name}",
            type="primary" if st.session_state.page==page_name else "secondary"
        ):
            st.session_state.page=page_name
            st.rerun()
    st.divider()
    if st.button("🚪 Logout",use_container_width=True,key="logout_button"):
        for key,value in DEFAULT_SESSION.items():
            st.session_state[key]=value
        st.rerun()

# ============================================================
# DASHBOARD
# ============================================================
if st.session_state.page=="Dashboard":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🏠 Dashboard</div>
        <div class="main-subtitle">Credit risk assessment overview and system activity</div>
    </div>""",unsafe_allow_html=True)
    st.markdown(
        f'<div class="welcome-banner">✅ &nbsp; Welcome back, {st.session_state.username}!</div>',
        unsafe_allow_html=True
    )

    try:
        summary=load_dashboard_summary() or {}
        recent=load_recent_predictions(8)
    except Exception:
        summary={};recent=pd.DataFrame()
        st.warning("Unable to load dashboard statistics from the database.")

    total=int(summary.get("total_predictions",0) or 0)
    high=int(summary.get("high_risk",0) or 0)
    low=int(summary.get("low_risk",0) or 0)
    avg=safe_float(summary.get("average_risk",0))
    if avg>1: avg/=100

    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card metric-blue"><div class="metric-title">📋 &nbsp; Total Predictions</div><div class="metric-value">{total}</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card metric-red"><div class="metric-title">🔴 &nbsp; High Risk</div><div class="metric-value">{high}</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card metric-green"><div class="metric-title">🟢 &nbsp; Low Risk</div><div class="metric-value">{low}</div></div>',unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card metric-purple"><div class="metric-title">📊 &nbsp; Average Risk</div><div class="metric-value">{avg*100:.1f}%</div></div>',unsafe_allow_html=True)

    if total==0:
        st.info("No prediction records are available yet. Start by creating a risk prediction.")

    chart1,chart2=st.columns(2)
    with chart1:
        st.markdown('<div class="content-card"><div class="card-title">Risk Distribution</div></div>',unsafe_allow_html=True)
        fig=make_risk_donut(low,high)
        st.pyplot(fig,clear_figure=True,use_container_width=True);plt.close(fig)
        st.markdown(
            f'<div style="text-align:center;color:#e2e8f0;font-weight:700">🟢 Low Risk ({low}) &nbsp;&nbsp;&nbsp; 🔴 High Risk ({high})</div>',
            unsafe_allow_html=True
        )
    with chart2:
        st.markdown('<div class="content-card"><div class="card-title">Recent Risk Probability</div></div>',unsafe_allow_html=True)
        fig=make_probability_histogram(recent)
        st.pyplot(fig,clear_figure=True,use_container_width=True);plt.close(fig)

    st.markdown('<div class="section-title">🕘 Recent Predictions</div>',unsafe_allow_html=True)
    render_recent_predictions(recent)

# ============================================================
# RISK PREDICTION
# ============================================================
elif st.session_state.page=="Risk Prediction":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🔍 Credit Risk Assessment</div>
        <div class="main-subtitle">Enter applicant information to evaluate the probability of serious delinquency.</div>
    </div>""",unsafe_allow_html=True)

    st.markdown('<div class="section-title">👤 Personal Information</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-description">Basic applicant information</div>',unsafe_allow_html=True)
    p1,p2,p3=st.columns(3)
    with p1:
        age=st.number_input("Age",18,100,int(defaults["age_median"]))
    with p2:
        monthly_income=st.number_input("Monthly Income ($)",min_value=0,value=int(defaults["MonthlyIncome_median"]),step=100)
    with p3:
        dependents=st.number_input("Number of Dependents",0,10,int(defaults["NumberOfDependents_mode"]))

    st.divider()
    st.markdown('<div class="section-title">💳 Credit Information</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-description">Applicant credit usage and loan information</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: debt_ratio=st.number_input("Debt Ratio",0.0,5.0,.3,.01)
    with c2: revolving_util=st.number_input("Revolving Utilization",0.0,2.0,.3,.01)
    with c3: open_credit_lines=st.number_input("Open Credit Lines / Loans",0,50,8)
    with c4: real_estate_loans=st.number_input("Real Estate Loans",0,20,1)

    st.divider()
    st.markdown('<div class="section-title">⚠️ Payment History</div>',unsafe_allow_html=True)
    h1,h2,h3=st.columns(3)
    with h1: late_30_59=st.number_input("30–59 Days Past Due",0,20,0)
    with h2: late_60_89=st.number_input("60–89 Days Past Due",0,20,0)
    with h3: late_90_plus=st.number_input("90+ Days Past Due",0,20,0)

    predict=st.button("🔍 Assess Credit Risk",type="primary",use_container_width=True)

    if predict:
        with st.spinner("Analyzing applicant credit profile..."):
            total_past_due=late_30_59+late_60_89+late_90_plus
            estimated_monthly_debt=debt_ratio*monthly_income
            income_per_dependent=monthly_income/(dependents+1)
            credit_lines_per_age_year=open_credit_lines/(age-17) if age>17 else 0
            has_real_estate_loan=1 if real_estate_loans>0 else 0
            input_dict={
                "RevolvingUtilizationOfUnsecuredLines":revolving_util,
                "age":age,
                "NumberOfTime30-59DaysPastDueNotWorse":late_30_59,
                "DebtRatio":debt_ratio,
                "MonthlyIncome":monthly_income,
                "NumberOfOpenCreditLinesAndLoans":open_credit_lines,
                "NumberOfTimes90DaysLate":late_90_plus,
                "NumberRealEstateLoansOrLines":real_estate_loans,
                "NumberOfTime60-89DaysPastDueNotWorse":late_60_89,
                "NumberOfDependents":dependents,
                "NumberOfTime30-59DaysPastDueNotWorse_was_anomaly":0,
                "NumberOfTimes90DaysLate_was_anomaly":0,
                "NumberOfTime60-89DaysPastDueNotWorse_was_anomaly":0,
                "MonthlyIncome_was_missing":0,
                "TotalPastDueIncidents":total_past_due,
                "EstimatedMonthlyDebtPayment":estimated_monthly_debt,
                "IncomePerDependent":income_per_dependent,
                "CreditLinesPerAgeYear":credit_lines_per_age_year,
                "HasRealEstateLoan":has_real_estate_loan,
            }
            try:
                input_df=pd.DataFrame([input_dict])[feature_columns]
                proba=float(model.predict_proba(input_df)[0][1])
                prediction=int(model.predict(input_df)[0])

                # SHAP is calculated once during prediction and then reused.
                explainer=load_shap_explainer()
                shap_values=explainer.shap_values(input_df)

                base_value=explainer.expected_value
                if isinstance(base_value,np.ndarray):
                    base_value=base_value.reshape(-1)[0]
                base_value=float(base_value)

                # Generate the waterfall image now so the Explainability page
                # opens immediately after navigation.
                shap_plot=build_shap_waterfall_png(
                    tuple(np.asarray(shap_values[0]).reshape(-1).tolist()),
                    tuple(input_df.iloc[0].values.tolist()),
                    tuple(feature_columns),
                    base_value,
                )

                st.session_state.prediction_done=True
                st.session_state.latest_probability=proba
                st.session_state.latest_prediction=prediction
                st.session_state.latest_input_df=input_df
                st.session_state.latest_input_dict=input_dict
                st.session_state.latest_shap_values=shap_values
                st.session_state.latest_shap_plot=shap_plot

                try:
                    db.insert_prediction(DB_CONFIG,input_dict,proba,prediction,st.session_state.username)
                    clear_database_caches()
                except Exception:
                    st.warning("Prediction completed, but the result could not be saved to prediction history.")
                st.success("Credit risk assessment completed successfully.")
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")

    if st.session_state.prediction_done:
        proba=st.session_state.latest_probability
        prediction=st.session_state.latest_prediction
        st.divider()
        st.markdown('<div class="section-title">📊 Risk Assessment Result</div>',unsafe_allow_html=True)
        result1,result2=st.columns([1,2])
        with result1:
            if prediction==1:
                st.markdown(f'<div class="risk-high"><div style="font-size:3rem">🔴</div><div class="risk-label">HIGH CREDIT RISK</div><div class="risk-percentage">{proba*100:.1f}%</div><div>Default Probability</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low"><div style="font-size:3rem">🟢</div><div class="risk-label">LOW CREDIT RISK</div><div class="risk-percentage">{proba*100:.1f}%</div><div>Default Probability</div></div>',unsafe_allow_html=True)
        with result2:
            st.markdown('<div class="section-title">Risk Interpretation</div>',unsafe_allow_html=True)
            st.progress(min(float(proba),1.0))
            if proba<.20: st.success("Low risk: the applicant's predicted default probability is relatively low.")
            elif proba<.50: st.warning("Moderate risk: the applicant may require additional review.")
            else: st.error("High risk: the model identified strong indicators associated with default.")
            r1,r2,r3=st.columns(3)
            r1.metric("Risk Score",f"{proba*100:.1f}%")
            r2.metric("Model","XGBoost")
            r3.metric("Explanation","SHAP")
        st.info("💡 Use the **Explainability** section from the sidebar to understand which factors influenced this prediction.")

# ============================================================
# EXPLAINABILITY
# ============================================================
elif st.session_state.page=="Explainability":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🧠 Model Explainability</div>
        <div class="main-subtitle">Understand why the model produced its prediction.</div>
    </div>""",unsafe_allow_html=True)

    if not st.session_state.prediction_done:
        st.info("No prediction is available yet. Go to **Risk Prediction** and assess an applicant first.")
    else:
        proba=st.session_state.latest_probability
        input_df=st.session_state.latest_input_df
        shap_values=st.session_state.latest_shap_values
        prediction=st.session_state.latest_prediction

        if prediction==1: st.error(f"Current prediction: 🔴 High Risk ({proba*100:.1f}%)")
        else: st.success(f"Current prediction: 🟢 Low Risk ({proba*100:.1f}%)")

        st.markdown("""
        ### How SHAP Works
        SHAP (SHapley Additive exPlanations) shows how individual features influence the model's prediction.

        **🔴 Red** → increases predicted risk

        **🔵 Blue** → decreases predicted risk
        """)
        st.divider()
        st.subheader("Feature Contribution")

        # The plot is prepared during prediction and stored in session state.
        # This avoids recalculating SHAP/waterfall every time the page opens.
        shap_plot=st.session_state.get("latest_shap_plot")

        if shap_plot:
            st.image(shap_plot, use_container_width=True)
        else:
            # Compatibility fallback for a session created before this
            # performance optimization.
            try:
                explainer=load_shap_explainer()
                base_value=explainer.expected_value
                if isinstance(base_value,np.ndarray):
                    base_value=base_value.reshape(-1)[0]
                fallback_plot=build_shap_waterfall_png(
                    tuple(np.asarray(shap_values[0]).reshape(-1).tolist()),
                    tuple(input_df.iloc[0].values.tolist()),
                    tuple(feature_columns),
                    float(base_value),
                )
                st.session_state.latest_shap_plot=fallback_plot
                st.image(fallback_plot, use_container_width=True)
            except Exception as exc:
                st.warning(f"SHAP waterfall plot could not be displayed: {exc}")

        st.subheader("Top Contributing Factors")
        shap_array=np.asarray(shap_values[0])
        shap_df=pd.DataFrame({
            "Feature":feature_columns,
            "SHAP Value":shap_array,
            "Applicant Value":input_df.iloc[0].values
        })
        shap_df["Impact"]=shap_df["SHAP Value"].apply(
            lambda x:"🔴 Increases Risk" if x>0 else "🔵 Decreases Risk"
        )
        shap_df=shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index)
        st.dataframe(
            shap_df.head(10)[["Feature","Applicant Value","Impact","SHAP Value"]],
            use_container_width=True,hide_index=True
        )

# ============================================================
# ANALYTICS
# ============================================================
elif st.session_state.page=="Analytics":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">📊 Prediction Analytics</div>
        <div class="main-subtitle">Historical credit risk prediction activity.</div>
    </div>""",unsafe_allow_html=True)

    # Fast path: aggregate statistics + recent records only.
    try:
        summary=load_dashboard_summary() or {}
        recent_df=prepare_recent_dataframe(load_analytics_recent(100))
    except Exception:
        summary={}
        recent_df=pd.DataFrame()
        st.error("Unable to retrieve analytics data.")

    total=int(summary.get("total_predictions",0) or 0)
    high_count=int(summary.get("high_risk",0) or 0)
    low_count=int(summary.get("low_risk",0) or 0)
    avg=safe_float(summary.get("average_risk",0))
    if avg>1:
        avg/=100

    if total==0:
        st.info("No prediction records are available.")
    else:
        a1,a2,a3,a4=st.columns(4)
        with a1:
            st.markdown(f'<div class="metric-card metric-blue"><div class="metric-title">📋 &nbsp; Total Predictions</div><div class="metric-value">{total}</div></div>',unsafe_allow_html=True)
        with a2:
            st.markdown(f'<div class="metric-card metric-red"><div class="metric-title">🔴 &nbsp; High Risk</div><div class="metric-value">{high_count}</div></div>',unsafe_allow_html=True)
        with a3:
            st.markdown(f'<div class="metric-card metric-green"><div class="metric-title">🟢 &nbsp; Low Risk</div><div class="metric-value">{low_count}</div></div>',unsafe_allow_html=True)
        with a4:
            st.markdown(f'<div class="metric-card metric-purple"><div class="metric-title">📊 &nbsp; Average Risk</div><div class="metric-value">{avg*100:.1f}%</div></div>',unsafe_allow_html=True)

        st.caption("Analytics charts use the most recent 100 predictions for faster loading.")

        st.divider()

        # Use the aggregate counts for the donut so it represents the
        # complete database without loading the complete history.
        ch1,ch2=st.columns(2)

        with ch1:
            st.subheader("Risk Distribution")
            fig=make_risk_donut(low_count,high_count)
            st.pyplot(fig,clear_figure=True,use_container_width=True)
            plt.close(fig)

        with ch2:
            st.subheader("Recent Risk Probability")

            probability_values=pd.to_numeric(
                recent_df.get("risk_probability",pd.Series(dtype=float)),
                errors="coerce"
            ).dropna()

            if probability_values.empty:
                st.info("No recent probability records available.")
            else:
                values=probability_values.to_numpy(dtype=float)
                values=np.where(values<=1,values*100,values)

                fig,ax=plt.subplots(figsize=(5.2,4.2))
                ax.hist(
                    values,
                    bins=10,
                    edgecolor="#0f1724",
                    color="#2f80ed"
                )
                ax.set_xlabel("Default Probability (%)")
                ax.set_ylabel("Number of Predictions")
                ax.set_facecolor("#111a28")
                fig.patch.set_facecolor("#111a28")
                ax.tick_params(colors="#e2e8f0")
                for spine in ax.spines.values():
                    spine.set_color("#475569")
                ax.grid(
                    axis="y",
                    color="#334155",
                    linestyle="--",
                    alpha=.7
                )
                plt.tight_layout()
                st.pyplot(fig,clear_figure=True,use_container_width=True)
                plt.close(fig)

        st.divider()
        st.subheader("📋 Recent Prediction History")

        if recent_df.empty:
            st.info("No recent prediction records available.")
        else:
            render_recent_predictions(recent_df)

        st.divider()

        # IMPORTANT:
        # st.expander does not lazily execute its body in Streamlit.
        # Therefore a checkbox is used so the full DB query happens only
        # when the user explicitly requests it.
        load_full_history=st.checkbox(
            "📂 Load Full Prediction History",
            help="The complete MySQL history is loaded only when this option is selected."
        )

        if load_full_history:
            with st.spinner("Loading complete prediction history..."):
                try:
                    log_df=load_prediction_history()
                except Exception:
                    log_df=pd.DataFrame()
                    st.error("Unable to retrieve complete prediction history.")

            if log_df.empty:
                st.info("No prediction records are available.")
            else:
                st.dataframe(
                    log_df.sort_values(
                        "prediction_timestamp",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()
        st.warning(
            "Clearing prediction history permanently removes all stored prediction records."
        )

        if st.button("🗑️ Clear Prediction History"):
            try:
                db.clear_predictions(DB_CONFIG)
                clear_database_caches()
                st.success("Prediction history cleared successfully.")
                st.rerun()
            except Exception:
                st.error("Unable to clear prediction history.")

# MODEL INFORMATION
# ============================================================
elif st.session_state.page=="Model Information":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🤖 Model Information</div>
        <div class="main-subtitle">Machine learning architecture and technology stack.</div>
    </div>""",unsafe_allow_html=True)
    st.subheader("Machine Learning Model")
    m1,m2,m3=st.columns(3)
    with m1:
        st.markdown('<div class="info-card"><b>Algorithm</b><h3>XGBoost Classifier</h3>Gradient boosting based binary classification model.</div>',unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="info-card"><b>Problem Type</b><h3>Binary Classification</h3>Predicts whether an applicant is at risk of serious delinquency.</div>',unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="info-card"><b>Explainability</b><h3>SHAP</h3>Provides feature-level explanations for model predictions.</div>',unsafe_allow_html=True)
    st.divider()
    st.subheader("🔄 Prediction Pipeline")
    pipeline=["Applicant Data","Feature Engineering","XGBoost Model","Risk Probability","Risk Classification","SHAP Explanation","MySQL Storage"]
    cols=st.columns(len(pipeline))
    for i,step in enumerate(pipeline):
        with cols[i]:
            st.markdown(f'<div class="info-card" style="text-align:center;min-height:100px"><b>{i+1}</b><br><br>{step}</div>',unsafe_allow_html=True)
    st.divider()
    st.subheader("🛠️ Technology Stack")
    t1,t2,t3,t4=st.columns(4)
    t1.markdown("### 🐍 Python");t1.caption("Application & ML")
    t2.markdown("### 📊 XGBoost");t2.caption("Prediction Model")
    t3.markdown("### 🧠 SHAP");t3.caption("Explainability")
    t4.markdown("### 🗄️ MySQL");t4.caption("Prediction Storage")
    st.divider()
    st.subheader("📦 Project Artifacts")
    artifacts=pd.DataFrame({
        "File":["model.json","feature_columns.json","defaults.json","dataset_averages.json"],
        "Purpose":["Trained XGBoost model","Model feature ordering","Default input values","Dataset reference values"]
    })
    st.dataframe(artifacts,use_container_width=True,hide_index=True)

# ============================================================
# ABOUT
# ============================================================
elif st.session_state.page=="About":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">ℹ️ About the Project</div>
        <div class="main-subtitle">Credit Risk Prediction System</div>
    </div>""",unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Project Objective
    The **Credit Risk Prediction System** estimates the probability that a loan applicant may experience serious credit delinquency.

    The system combines machine learning prediction, explainability, secure authentication, database storage, and an interactive web interface.

    ---

    ### 🔐 Security
    The application uses an administrator login system. Passwords are stored using **bcrypt hashing** rather than plain-text passwords.

    ---

    ### 🧠 Explainable Predictions
    The application uses **SHAP** to show which applicant features contributed to the model's prediction.

    ---

    ### 🗄️ Database
    Prediction results are stored in a MySQL database, including applicant information, predicted risk, probability, timestamp, and authorized user.

    ---

    ### 🚀 Deployment
    The application is built using Streamlit and can be deployed as a web-based machine learning system.

    ---

    ### 📌 Project Title
    **Credit Risk Prediction System**
    """)
    st.divider()
    st.markdown('<div class="footer">🛡️ Credit Risk Prediction System<br>XGBoost • SHAP • MySQL • Streamlit</div>',unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Credit Risk Prediction System &nbsp;•&nbsp; Secure ML-Based Credit Risk Assessment</div>',
    unsafe_allow_html=True
)