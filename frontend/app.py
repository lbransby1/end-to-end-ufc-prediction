import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from comparison_table import render_comparison_table


# Set wide layout and dashboard title
st.set_page_config(layout="wide", page_title="UFC Fight Predictor Dashboard")
st.markdown(
    """
    <h3 style='text-align:center; color:white; margin:0; padding:0;'>
         UFC Fight Predictor Dashboard
    </h3>
    <h5 style='text-align:center; font-weight:normal; color:#555;'>
        Compare fighters, break down stats, and predict outcomes with <b>Machine Learning</b>
    </h5>
    <p style='text-align:center;'>
        <a href="https://github.com/lbransby1/MMA-Metrics--Machine-Learning-Dashboard-for-UFC-Predictions" target="_blank" style="text-decoration:none; color:#1f77b4; font-size:16px;">
            🔗 View the GitHub here
        </a>
    </p><hr>
    """,
    unsafe_allow_html=True
)





# Load and clean fighter data
fighters_df = pd.read_csv("processed_data/fighter_averages.csv")
fighters_df["Name"] = fighters_df["Name"].astype(str)
fighter_names = sorted(fighters_df["Name"].dropna().unique())

# Load fight styles
styles_df = pd.read_csv("processed_data/fight_style_descriptions.csv")
fighters_df = fighters_df.merge(styles_df, on="Name", how="left")

# Convert DOB to Age
fighters_df["DOB"] = pd.to_datetime(fighters_df["DOB"], errors='coerce')
today = pd.Timestamp.today()
fighters_df["Age"] = fighters_df["DOB"].apply(lambda dob: (today - dob).days // 365 if pd.notnull(dob) else np.nan)

# Averaged stats only
avg_stats = [col for col in fighters_df.columns if (
    "PerMin" in col or "Per15Min" in col or "AccuracyPct" in col or "DefencePct" in col or "Pct" in col)
    and col != "KnockdownPct"]

@st.cache_data(show_spinner=False)
def get_ufc_image(fighter_name):
    base_url = "https://www.ufc.com/athlete/"
    slug = fighter_name.lower().replace(" ", "-")
    url = base_url + slug
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = soup.find('img', class_='hero-profile__image')
        return img_tag['src'] if img_tag else None
    except Exception:
        return None

# Default fighters
f1_default = "Alexander Volkanovski"
f2_default = "Max Holloway"

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Select Fighters")
    fighter_1 = st.selectbox("Red Corner", fighter_names, index=fighter_names.index(f1_default))
    fighter_2 = st.selectbox("Blue Corner", fighter_names, index=fighter_names.index(f2_default))

    with st.expander("Developer Options", expanded=False):
        debug_mode = st.checkbox("Enable Dev Mode")


    if debug_mode:
        st.sidebar.success("Developer mode is ON")


def display_fighter_card(name, corner_color, image_width=200):
    import streamlit as st
    fighter_row = fighters_df[fighters_df["Name"] == name].iloc[0]
    # Map color to label for display
    color_label = f"{name}" 

    # Extract data
    name = fighter_row["Name"]
    image_url = get_ufc_image(name)  # Fallback if missing
    details = {
        "Age": fighter_row.get("Age", "N/A"),
        "Height": fighter_row.get("Height", "N/A"),
        "Reach": fighter_row.get("Reach", "N/A"),
        "Weight": fighter_row.get("Weight", "N/A"),
        "Stance": fighter_row.get("Stance", "N/A"),
        "Wins": fighter_row.get("Wins", "N/A"),
        "Losses": fighter_row.get("Losses", "N/A")
    }
    font_size = 16

    # Corner label with spacing
    st.markdown(f"<h3 style='margin-bottom:12px; padding-left:10px'>{color_label}</h3>", unsafe_allow_html=True)

    # Create two columns: one for image, one for stats
    col1, col2 = st.columns([1, 1.5], gap="medium")  # Add medium gap between cols

    with col1:
        st.image(image_url, width=image_width, caption=name)

    with col2:
        # Use HTML for font size + spacing
        stats_html = "<div style='line-height:1.8'>"  # add vertical spacing between rows
        for stat_name, stat_value in details.items():
            stats_html += f"<div style='font-size:{font_size}px; margin-bottom:6px'>{stat_name}: <b>{stat_value}</b></div>"
        stats_html += "</div>"
        st.markdown(stats_html, unsafe_allow_html=True)

    # Extra spacing between sections
    st.markdown(f"<p style='font-size:{font_size}px; margin-top:0px'><b>Style</b>: {fighter_row.get('Style','N/a')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{font_size}px; margin-top:0px'><b>Strengths</b>: {fighter_row.get('Strengths','N/a')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:{font_size}px; margin-top:4px'><b>Weaknesses</b>: {fighter_row.get('Weaknesses','N/a')}</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    display_fighter_card(fighter_1, corner_color="red")
with col2:
    display_fighter_card(fighter_2, corner_color="blue")

# Variance Weights
variance_weights = {
    'PerMin': 2.0, 'Per15Min': 1.5, 'Pct': 1.0, 'AccuracyPct': 1.0, 'DefencePct': 1.0
}

def get_weight(stat):
    for key, weight in variance_weights.items():
        if key in stat:
            return weight
    return 1.0

def compute_percentile_ratings(df, stats, invert_stats=None):
    if invert_stats is None:
        invert_stats = []
    ratings = pd.DataFrame(index=df.index)
    for stat in stats:
        percentiles = df[stat].rank(pct=True)
        if stat in invert_stats:
            percentiles = 1 - percentiles
        ratings[stat] = (percentiles * 10).round(2)
    return ratings

roster_ratings = compute_percentile_ratings(fighters_df, avg_stats, invert_stats=["StrikesAbsorbedPerMin"])

f1_row = fighters_df[fighters_df["Name"] == fighter_1].index[0]
f2_row = fighters_df[fighters_df["Name"] == fighter_2].index[0]

f1_ratings = roster_ratings.loc[f1_row]
f2_ratings = roster_ratings.loc[f2_row]

f1_raw = fighters_df.loc[f1_row, avg_stats]
f2_raw = fighters_df.loc[f2_row, avg_stats]
comparison_df = pd.DataFrame({fighter_1: f1_raw.values, "Stat": avg_stats, fighter_2: f2_raw.values})
# Round all numeric columns to 2 decimal places
# Only round numeric columns, keep 'Stat' as-is
numeric_cols = [c for c in comparison_df.columns if c != "Stat"]
comparison_df[numeric_cols] = comparison_df[numeric_cols].applymap(lambda x: f"{x:.2f}")


# Stat Highlighting

def highlight_row(row):
    if len(row) < 3 or pd.isna(row[1]) or pd.isna(row[2]):
        return [''] * len(row)
    val1, stat, val2 = row[0], row[1], row[2]
    return [
        'background-color: #006400' if val1 > val2 else 'background-color: #8B0000' if val1 < val2 else '',
        '',
        'background-color: #006400' if val2 > val1 else 'background-color: #8B0000' if val2 < val1 else ''
    ]

st.markdown("---")

# Create two columns
col_table, col_radar = st.columns([1, 1])  # Equal width, can adjust ratio if needed

with col_table:
    st.markdown("<h3 style='text-align:center;'>Stat Comparison</h3><br><br>", unsafe_allow_html=True)

    render_comparison_table(comparison_df, highlight_row)  # Your styled comparison table
with col_radar:
    st.markdown("<h3 style='text-align:center;'>Performance Radar</h3>", unsafe_allow_html=True)

    # 1. Ensure we actually have stats to plot
    if not avg_stats:
        st.warning("No averaged stats found for radar chart.")
    else:
        # 2. Get the ratings for the two selected fighters
        try:
            f1_data = roster_ratings.loc[f1_row, avg_stats].tolist()
            f2_data = roster_ratings.loc[f2_row, avg_stats].tolist()
            
            # Radar charts need to "close the loop" by repeating the first value
            radar_stats = avg_stats + [avg_stats[0]]
            f1_data = f1_data + [f1_data[0]]
            f2_data = f2_data + [f2_data[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=f1_data, theta=radar_stats, fill='toself',
                name=fighter_1, line=dict(color='crimson')
            ))
            fig.add_trace(go.Scatterpolar(
                r=f2_data, theta=radar_stats, fill='toself',
                name=fighter_2, line=dict(color='royalblue')
            ))
            
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 10]),
                ),
                showlegend=True,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating radar: {e}")

def swap_averaged_all(models, X_orig, X_swapped, y_true=None, fighters=None):
    """
    Returns a dictionary of results for each model, including:
    - pre-swap probabilities
    - swap-averaged probabilities
    - predicted winner
    - accuracy (if y_true provided)
    """
    results = {}
    for name, model in models.items():
        # Predict probabilities
        orig_probs = model.predict_proba(X_orig)
        swap_probs = model.predict_proba(X_swapped)
        swap_probs_corrected = swap_probs[:, [1, 0]]  # flip to align with original

        # Swap-averaged probabilities
        final_probs = (orig_probs + swap_probs_corrected) / 2
        print(name, "og prob", orig_probs, "swap prob:", swap_probs, final_probs)
        # Single fight winner
        if fighters and final_probs.shape[0] == 1:
            f1, f2 = fighters
            winner = f1 if final_probs[0][0] > final_probs[0][1] else f2
            results[name] = {
                "PreSwapProbs": orig_probs[0],
                "Probs": final_probs[0],
                "Winner": winner
            }
        else:
            # Dataset
            y_pred_final = np.argmax(final_probs, axis=1)
            acc = None
            if y_true is not None:
                acc = (y_pred_final == y_true).mean()
            results[name] = {
                "PreSwapProbs": orig_probs,
                "Probs": final_probs,
                "Accuracy": acc
            }
    return results

# fighter_1 = "Jack Della Maddalena"
# fighter_2 = "Ilia Topuria"

# --- Save the feature order from training ---

def create_features_from_df(f1, f2, df):
    red = df[df["Name"] == f1].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
    blue = df[df["Name"] == f2].drop(columns=["Name", "DOB"], errors='ignore').iloc[0]
    red.index = ["red_" + col for col in red.index]
    blue.index = ["blue_" + col for col in blue.index]
    return pd.concat([red, blue]).to_frame().T
    

# --- API Prediction ---
st.markdown(
    """
    <style>
    .stButton > button {
        transform: scale(1);  /* make button + text 1.5x bigger */
        font-weight: bold;
        width: 100%;
        margin: 0 auto;  /* centers the button */
        display: block;  /* required for margin auto to work */
    }
    </style>
    """,
    unsafe_allow_html=True
)

predict_button = st.button("Predict")

if predict_button:
    with st.spinner('Asking the API for predictions...'):
        # 1. Prepare the payload
        api_payload = {
            "fighter_red": fighter_1,
            "fighter_blue": fighter_2
        }
        
        try:
            # 2. Call the FastAPI backend inside the Docker network
            response = requests.post("http://backend:8000/predict", json=api_payload)
            response.raise_for_status()
            
            # 3. Parse the result
            prediction_data = response.json()
            winner = prediction_data["winner"]
            confidence = prediction_data["confidence"]
            rf_probs = prediction_data["inference_results"]["RandomForest"]
            
            # 4. Display the results using your awesome custom HTML
            st.markdown(
                f"""
                <div style="
                    background-color:#222;
                    color:#fff;
                    border: 2px solid #444;
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                    margin-bottom: 20px;">
                    🏆 Winner Prediction: <span style="color:#3fcf5f">{winner}</span><br>
                    Confidence: {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show the underlying model probabilities
            st.info(f"**Random Forest Breakdown:** 🔴 Red {rf_probs['red_win_prob']:.2%} | 🔵 Blue {rf_probs['blue_win_prob']:.2%}")
            
        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not connect to the Backend API! Is the Docker container running?")
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")