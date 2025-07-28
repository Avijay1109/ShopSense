# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="ShopSense", layout="wide")

# Hide 'Press Enter to submit' UI hint on number_input fields
st.markdown("""
    <style>
    .stNumberInput > div > div > input:focus-visible {
        outline: none !important;
    }
    .stNumberInput .e1b2p2ww10 {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Suppress Enter key to prevent premature form submission
st.markdown("""
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && event.target.tagName === 'INPUT') {
            event.preventDefault();
        }
    });
    </script>
""", unsafe_allow_html=True)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #F2F2F4;
        color: #000000;
    }
    .main {
        background-color: #FFFFFF;
    }
    h1 {
        text-align: center;
        color: #9D2235;
        font-size: 48px;
        font-weight: bold;
    }
    .stButton button {
        background-color: #9D2235;
        color: #FFFFFF;
        font-weight: bold;
    }
    .stDownloadButton button {
        background-color: #2B5269;
        color: #FFFFFF;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; font-size:60px; color:#9D2235;'>ShopSense</h1>", unsafe_allow_html=True)
st.image("Images/Website Banner.png", use_container_width=True)

st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)

# CTA state setup
if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False
if "upload_mode" not in st.session_state:
    st.session_state.upload_mode = False
if "view_insights" not in st.session_state:
    st.session_state.view_insights = False

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    if st.button("Enter Manually"):
        st.session_state.manual_mode = True
        st.session_state.upload_mode = False
        st.session_state.view_insights = False
with col2:
    if st.button("View Insights"):
        st.session_state.view_insights = True
        st.session_state.manual_mode = False
        st.session_state.upload_mode = False

# --- Prepare Data and Model ---
df = pd.read_csv("Data/online_shoppers_intention.csv")
df_model = df.copy()

label_encoders = {}
for col in df_model.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Final features used in model
final_features = [
    'PageValues', 'ExitRates', 'ProductRelated_Duration', 'ProductRelated',
    'Administrative_Duration', 'Administrative', 'BounceRates', 'EngagementScore'
]

# Feature display names and tooltips
feature_labels = {
    'PageValues': ('Page Values', "Average monetary value a page contributes to a user's session"),
    'ExitRates': ('Exit Rates', 'The percentage of website visits that end on a specific web page'),
    'BounceRates': ('Bounce Rates', 'The percentage of single-page sessions on your website'),
    'EngagementScore': ('Engagement Score', '(Page Values/Exit Rates + 0.01)'),
    'ProductRelated_Duration': ('Time Spent on Product Pages', ''),
    'ProductRelated': ('No. of Product Pages Visited', ''),
    'Administrative_Duration': ('Time Spent on Administrative Pages', ''),
    'Administrative': ('No. of Administrative Pages Visited', '')
}

# Calculate engagement score
df_model['EngagementScore'] = df_model['PageValues'] / (df_model['ExitRates'] + 0.01)
X = df_model[final_features]
y = df_model['Revenue']
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)

# --- Manual Entry Form ---
if st.session_state.manual_mode:
    with st.form("prediction_form"):
        input_values = {}
        for feature in final_features:
            if feature == "EngagementScore":
                continue  # Skip manual entry for auto-calculated field
            label, helptext = feature_labels.get(feature, (feature, ''))
            input_values[feature] = st.text_input(label, placeholder="Enter a positive number", help=helptext)

        submitted = st.form_submit_button("Predict Purchase Intent")

        if submitted:
            try:
                # Check for invalid values (empty or negative)
                invalid = False
                for key, val in input_values.items():
                    try:
                        float_val = float(val)
                        if float_val < 0:
                            invalid = True
                            break
                        input_values[key] = float_val
                    except:
                        invalid = True
                        break

                if invalid:
                    st.error("Input not valid. All values must be non-negative numbers.")
                else:
                    # Auto-calculate engagement score
                    input_values = {k: float(v) for k, v in input_values.items()}
                    input_values['EngagementScore'] = input_values['PageValues'] / (input_values['ExitRates'] + 0.01)

                    input_df = pd.DataFrame([input_values])
                    input_df = input_df[final_features]
                    proba = model.predict_proba(input_df)[0][1]  # probability of purchase
                    prediction = int(proba >= 0.70)
                    prob = proba
                    st.markdown("""
                        <style>
                        .slide-in {
                            animation: slideIn 0.6s ease-out forwards;
                            transform: translateY(30px);
                            opacity: 0;
                        }
                        @keyframes slideIn {
                            to {
                                transform: translateY(0);
                                opacity: 1;
                            }
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    if prediction:
                        st.markdown(f"""
                            <div class='slide-in' style='padding: 20px; background-color: #DFF0D8; color: #3C763D; border-radius: 8px; text-align: center;'>
                                <h3 style='margin-bottom: 10px;'>✅ Likely to Purchase</h3>
                                <p style='font-size: 18px;'>Confidence: {prob:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='slide-in' style='padding: 20px; background-color: #F2DEDE; color: #A94442; border-radius: 8px; text-align: center;'>
                                <h3 style='margin-bottom: 10px;'>❌ Not Likely to Purchase</h3>
                                <p style='font-size: 18px;'>Confidence: {prob:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- Insights View ---
if st.session_state.view_insights:
    st.subheader("Key Insights Dashboard")

    st.markdown("### **Fig 1 - Feature Correlation with Purchase Intent**")
    st.image("Images/Fig 1.png")
    st.caption("PageValues, ProductRelated activity, and ProductRelated_Duration are the strongest positive predictors of purchase intent, while ExitRates and BounceRates are negatively correlated. Most technical and categorical features show minimal correlation.")
    st.markdown("**Business Recommendation:**\nFocus on increasing product engagement and perceived page value by enriching product detail pages and highlighting promotions. Simultaneously, minimize friction points that elevate Exit and Bounce Rates—like slow load times or poor navigation—to strengthen conversion outcomes.")

    st.markdown("### **Fig 2 - Avg. Purchase Rate by Page Value Bin**")
    st.image("Images/Fig 2.png")
    st.caption("Sessions with higher Page Value strongly correlate with increased likelihood of purchase, with conversion rates rising from just 5.3% in the lowest bin (1–5) to 86.6% in the highest (100+).")
    st.markdown("**Business Recommendation:**\nInvest in strategies that increase Page Value—such as showcasing relevant promotions, improving product recommendations, or enhancing content depth—to significantly boost purchase intent and drive higher conversions.")

    st.markdown("### **Fig 3 - Avg. Purchase Rate by Exit Rate Bin**")
    st.image("Images/Fig 3.png")
    st.caption("Purchase rates sharply decline as Exit Rates increase—from 19.2% in the lowest bin (0–0.05) to less than 1% beyond 0.1—highlighting Exit Rate as a strong negative predictor of conversion.")
    st.markdown("**Business Recommendation:**\nPrioritize reducing Exit Rates through improved page flow, engaging content, and clearer call-to-actions—especially within the early part of the session—to retain potential buyers and improve conversion outcomes.")

    st.markdown("### **Fig 4 - Average Values of Key Features**")
    st.image("Images/Fig 4.png")
    st.caption("Product-related engagement—both in terms of time spent and pages viewed—dominates user behavior, with an average session time of over 1800 seconds and 48 product pages viewed. In contrast, friction metrics like Exit and Bounce Rates remain low on average.")
    st.markdown("**Business Recommendation:**\nDouble down on optimizing product page experiences through richer content, dynamic recommendations, and streamlined navigation. These high-engagement areas offer the greatest leverage for boosting conversions, while low friction rates should be maintained with intuitive site design and fast load speeds.")