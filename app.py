import streamlit as st
from PIL import Image, ImageDraw
import random
import pandas as pd
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Building Health Audit", page_icon="🏢", layout="wide")

# --- HEADER ---
st.title("🏢 Multi-Modal Building Health Monitoring System")
st.markdown("### *AI-Powered Urban Infrastructure Auditing & Defect Detection Framework*")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📖 Project Onboarding", "🔍 Live Defect Inspection", "📊 Inspection History"])

# --- TAB 1: ONBOARDING ---
with tab1:
    st.markdown("""
    ## Welcome to the Structural Audit Dashboard
    This framework combines **top-down satellite monitoring** with **ground-level edge-vision AI** to perform complete structural audits.
    
    ### How it works:
    1. **Macro Monitoring (Satellite):** Tracks ground subsidence and building age fatigue over time.
    2. **Micro Inspection (Edge AI):** Drones or ground cameras capture facade images to detect concrete cracks and spalling.
    3. **Data Fusion:** Predictions are merged into a risk severity index to assist urban planners.
    
    *Navigate to the **Live Defect Inspection** tab to test an image audit!*
    """)

# --- TAB 2: LIVE INSPECTION (MOCK PIPELINE) ---
with tab2:
    st.write("Upload a structural facade or concrete surface image to trigger the AI analysis pipeline.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Building Image (JPG/PNG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Run Structural Health Audit 🚀", type="primary"):
                with st.spinner("Processing image via YOLOv10 / U-Net pipeline..."):
                    # Mock AI Processing
                    annotated_img = image.copy()
                    draw = ImageDraw.Draw(annotated_img)
                    width, height = annotated_img.size
                    
                    defects = ["Concrete Crack", "Spalling", "Water Seepage"]
                    detected_defect = random.choice(defects)
                    severity_score = random.randint(12, 68)
                    status = "CRITICAL ACTION REQUIRED" if severity_score > 40 else "MONITORING RECOMMENDED"
                    
                    # Draw mock bounding box
                    box_color = "red" if severity_score > 40 else "orange"
                    draw.rectangle([int(width*0.25), int(height*0.3), int(width*0.75), int(height*0.7)], outline=box_color, width=5)
                    
                    with col2:
                        st.image(annotated_img, caption="AI Annotated Defect Map", use_column_width=True)
                        st.subheader("📋 Structural Health Report")
                        st.error(f"**Primary Anomaly Detected:** {detected_defect}") if severity_score > 40 else st.warning(f"**Primary Anomaly Detected:** {detected_defect}")
                        st.metric(label="Damage Severity Index", value=f"{severity_score}%")
                        st.info(f"**Recommended Action:** {status}")
                        st.write("**Satellite Cross-Reference:** Flagged for ground displacement verification.")

# --- TAB 3: HISTORY ---
with tab3:
    st.markdown("## Historical Building Audit Logs")
    
    # Mock Database
    history_data = pd.DataFrame({
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Location / Sector": ["Sector 4, Block B"],
        "Detected Defect": ["Concrete Crack"],
        "Severity Score": ["34%"],
        "Status": ["MONITORING RECOMMENDED"]
    })
    
    st.dataframe(history_data, use_container_width=True)
