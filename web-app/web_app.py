import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def load_knowledge_base():
    data = [
        {
            "name": "Tesseract 5", "type": "Standard OCR", "deployment": "Local", "input": "Page", 
            "modality": "Print", "cost": "Free", "barrier": "Medium", 
            "hallucination_risk": "Low", "infra_heavy": False, "trained": False,
            "license": "Open", "layout_flex": "Low", "lang_scope": "High", 
            "capex_heavy": False, "output_style": "Rigid", "throughput_mode": "Batch"
        },
        {
            "name": "ABBYY FineReader", "type": "Commercial OCR", "deployment": "Local", "input": "Page", 
            "modality": "Print", "cost": "Paid", "barrier": "Low", 
            "hallucination_risk": "Low", "infra_heavy": False, "trained": False,
            "license": "Closed", "layout_flex": "Medium", "lang_scope": "High", 
            "capex_heavy": False, "output_style": "Rigid", "throughput_mode": "Batch"
        },
        {
            "name": "PaddleOCR", "type": "Industrial OCR", "deployment": "Local", "input": "Page", 
            "modality": "Mixed", "cost": "Free", "barrier": "Medium", 
            "hallucination_risk": "Low", "infra_heavy": False, "trained": False,
            "license": "Open", "layout_flex": "Medium", "lang_scope": "Very High",
            "capex_heavy": False, "output_style": "Rigid", "throughput_mode": "Batch"

        },
        {
            "name": "Google ML Kit", "type": "Mobile SDK", "deployment": "Local", "input": "Page", 
            "modality": "Mixed", "cost": "Free", "barrier": "High", 
            "hallucination_risk": "Low", "infra_heavy": False, "trained": False,
            "license": "Closed", "layout_flex": "Low", "lang_scope": "Medium", 
            "capex_heavy": False, "output_style": "Rigid", "throughput_mode": "Stream"
        },
        {
        
            "name": "Kraken", "type": "Specialized HTR", "deployment": "Local", "input": "Line", 
            "modality": "Hand", "cost": "Free", "barrier": "High", 
            "hallucination_risk": "Low", "infra_heavy": True, "trained": True,
            "license": "Open", "layout_flex": "High", "lang_scope": "Low", 
            "capex_heavy": True, "output_style": "Faithful", "throughput_mode": "Batch"
        }, 
        {
            "name": "Calamari", "type": "Specialized HTR", "deployment": "Local", "input": "Line", 
            "modality": "Hand", "cost": "Free", "barrier": "High", 
            "hallucination_risk": "Low", "infra_heavy": True, "trained": True,
            "license": "Open", "layout_flex": "High", "lang_scope": "Low", 
            "capex_heavy": True, "output_style": "Faithful", "throughput_mode": "Batch"
        },
        {
            "name": "Transkribus", "type": "Specialized HTR", "deployment": "Cloud", "input": "Page", 
            "modality": "Mixed", "cost": "Paid", "barrier": "Low", 
            "hallucination_risk": "Low", "infra_heavy": False, "trained": True,
            "license": "Closed", "layout_flex": "High", "lang_scope": "High", 
            "capex_heavy": False, "output_style": "Faithful", "throughput_mode": "Batch"
        },
        {
            "name": "eScriptorium", "type": "Specialized HTR", "deployment": "Local", "input": "Page", 
            "modality": "Hand", "cost": "Free", "barrier": "Medium", 
            "hallucination_risk": "Low", "infra_heavy": True, "trained": True,
            "license": "Open", "layout_flex": "High", "lang_scope": "Low",
            "capex_heavy": True, "output_style": "Faithful", "throughput_mode": "Batch"
        },
    
        {
            "name": "TrOCR (Base)", "type": "Specialized HTR", "deployment": "Local", "input": "Line", 
            "modality": "Mixed", "cost": "Free", "barrier": "High", 
            "hallucination_risk": "Low", "infra_heavy": True, "trained": True,
            "license": "Open", "layout_flex": "Low", "lang_scope": "Medium",
            "capex_heavy": True, "output_style": "Faithful", "throughput_mode": "Batch"
        },

        {
            "name": "Claude 4.5 Opus", "type": "MLLMs", "deployment": "Cloud", "input": "Page", 
            "modality": "Mixed", "cost": "Paid", "barrier": "Low", 
            "hallucination_risk": "High", "infra_heavy": False, "trained": False,
            "license": "Closed", "layout_flex": "Very High", "lang_scope": "Very High",
            "capex_heavy": False, "output_style": "Fluent", "throughput_mode": "Stream"
        },
        {
            "name": "Gemini 2.5 Flash", "type": "MLLMs", "deployment": "Cloud", "input": "Page", 
            "modality": "Mixed", "cost": "Paid", "barrier": "Low", 
            "hallucination_risk": "High", "infra_heavy": False, "trained": False,
            "license": "Closed", "layout_flex": "Very High", "lang_scope": "Very High",
            "capex_heavy": False, "output_style": "Fluent", "throughput_mode": "Stream"
        },
        {
            "name": "Qwen2.5-VL", "type": "MLLMs", "deployment": "Local", "input": "Page", 
            "modality": "Mixed", "cost": "Free", "barrier": "High", 
            "hallucination_risk": "High", "infra_heavy": True, "trained": False,
            "license": "Open", "layout_flex": "Very High", "lang_scope": "High",
            "capex_heavy": True, "output_style": "Fluent", "throughput_mode": "Stream"
        },]
    return pd.DataFrame(data)


def calculate_suitability(df, inputs):
    scores = {}
    calc_df = df.copy()
    
    for index, tool in calc_df.iterrows():
        score = 0
        
        # type A : Project Requirements
    

        if tool['deployment'] == 'Cloud': score += (10 - inputs['privacy'])
        else: score += 10

        if tool['license'] == 'Closed': score += (10 - inputs['license'])
        else: score += 10
            
        if tool['trained']: score += (10 - inputs['urgency'])
        else: score += 10
            
        if tool['layout_flex'] in ['Low', 'Medium']:
            score += (10 - inputs['materiality']) if inputs['materiality'] > 5 else 10
        else: score += 10
            
        if tool['lang_scope'] in ['Low', 'Medium'] and not tool['trained']: score += (10 - inputs['language'])
        else: score += 10 

        if inputs['teleology'] >= 8: 
            if tool['output_style'] == 'Fluent': score += 0 
            elif tool['output_style'] == 'Faithful':
                score += 10
            elif tool['output_style'] == 'Rigid': 
                score += 3
            
            if tool['hallucination_risk'] == 'High':
                score -= 5

        elif inputs['teleology'] <= 2: 
            if tool['output_style'] == 'Fluent': score += 10
        
            elif tool['output_style'] == 'Faithful': 
                score += 9
            elif tool['output_style'] == 'Rigid': 
                score += 4

        else: score += 10
            
        if tool['throughput_mode'] == 'Stream': score += (10 - inputs['volume'])
        else: score += 10

        # type B: Institutional Resources
        if tool['barrier'] == 'High': score += inputs['expertise']
        elif tool['barrier'] == 'Medium': score += max(5, inputs['expertise'])
        else: score += 10
            
        if tool['infra_heavy']: score += inputs['hardware']
        else: score += 10
            
        if tool['trained']: score += inputs['gt_data']
        else: score += 10
            
        if tool['capex_heavy']: score += inputs['capex']
        else: score += 10

        scores[tool['name']] = score

    max_possible = 120 
    ranked_df = pd.DataFrame(list(scores.items()), columns=['Tool', 'Raw_Score'])
    ranked_df = ranked_df.merge(calc_df, left_on='Tool', right_on='name', how='left')
    ranked_df['Suitability'] = (ranked_df['Raw_Score'] / max_possible) * 100
    ranked_df = ranked_df.sort_values(by='Suitability', ascending=False)
    
    return ranked_df






st.set_page_config(page_title="GLAM ATR Selector", layout="wide", initial_sidebar_state="expanded")


st.title("GLAM ATR Decision Support Tool")
st.markdown("""
**A Multi-Criteria Evaluation Framework for Historical Text Recognition**
            
Align your project's technical requirements with institutional capabilities to identify the optimal digitization workflow.            
""")


# Stage 1 (Hard Conatraints) 
st.sidebar.header("Stage 1: Hard Constraints")
st.sidebar.info("Filter out tools that are impossible to use.")


material_type = st.sidebar.radio(
    "Material Modality",
    ("Pure Handwritten", "Pure Printed", "Mixed / Multimodal"),
    index=2,
    help="Select the dominant nature of your source documents."
)

constraint_local = st.sidebar.checkbox("Local Processing Only", value=False)
constraint_page = st.sidebar.checkbox("Full Page Input Only", value=False)
constraint_free = st.sidebar.checkbox("Open Source Only", value=False)
constraint_gui = st.sidebar.checkbox("GUI Required (No Code)", value=False)



tools_df = load_knowledge_base()
filtered_df = tools_df.copy()



# Logic for Material Modality
if material_type == "Pure Handwritten":
    filtered_df = filtered_df[filtered_df['modality'] != 'Print']
elif material_type == "Pure Printed":
    filtered_df = filtered_df[filtered_df['modality'] != 'Hand']
elif material_type == "Mixed / Multimodal":
    filtered_df = filtered_df[filtered_df['modality'] != 'Print']

# Logic for Checkboxes
if constraint_local: filtered_df = filtered_df[filtered_df['deployment'] == 'Local']
if constraint_page: filtered_df = filtered_df[filtered_df['input'] == 'Page']
if constraint_free: filtered_df = filtered_df[filtered_df['cost'] == 'Free']
if constraint_gui: filtered_df = filtered_df[filtered_df['barrier'] == 'Low']



st.sidebar.markdown("---")
delta_val = len(filtered_df)-len(tools_df)
st.sidebar.metric(
    "Remaining Candidates", 
    f"{len(filtered_df)}", 
    delta=f"{delta_val}" if delta_val != 0 else None,
    delta_color="off"
)

# Stagee 2 (Soft Constraints)
st.header("Stage 2: Soft Constraints & Preferences")

if len(filtered_df) == 0:
    st.error("No tools match your Stage 1 filters. Please relax the Hard Constraints in the sidebar.")
else:
    col_req, col_res = st.columns([1.2, 1]) 
    inputs = {}

    with col_req:
        st.subheader("A. Project Requirements (Strictness)")
        st.caption("10 = Very Strict / High Priority")
        
        
        inputs['privacy'] = st.slider("1. Privacy Sensitivity", 0, 10, 5, help="10 = Highly sensitive data.")
        inputs['license'] = st.slider("2. Open Source Preference", 0, 10, 5, help="10 = Must be open source.")
        inputs['materiality'] = st.slider("3. Material Complexity", 0, 10, 5, help="10 = Complex layouts (tables/margin).")
        inputs['language'] = st.slider("4. Language Rarity", 0, 10, 5, help="10 = Rare/Low-resource language.")
        inputs['teleology'] = st.slider(
            "5. Output Goal", 0, 10, 5, 
            help="0 = Data Mining/Reading (Fluent content); 10 = Archival/Diplomatic (Faithful representation)."
        )
        inputs['volume'] = st.slider(
            "6. Processing Volume", 0, 10, 5, 
            help="0 = Single Documents; 10 = Massive Archive (Batch processing preferred)."
        )
        inputs['urgency'] = st.slider("7. Time Urgency", 0, 10, 5, help="10 = Immediate results needed.")

    with col_res:
        st.subheader("B. Institutional Resources (Capabilities)")
        st.caption("10 = High Capability / Available")
        
        inputs['expertise'] = st.slider("8. Technical Expertise", 0, 10, 5, help="10 = Expert Developer.")
        inputs['hardware'] = st.slider("9. Hardware Infrastructure", 0, 10, 5, help="10 = High-end GPU Server.")
        inputs['gt_data'] = st.slider("10. Ground Truth Availability", 0, 10, 5, help="10 = Extensive Training Data.")
        inputs['capex'] = st.slider("11. CAPEX / Budget Availability", 0, 10, 5, help="10 = High budget for hardware/setup.")

   
    ranked_results = calculate_suitability(filtered_df, inputs)
    
    st.divider()
    
    st.subheader("Recommendation Analysis")
    
    r_col1, r_col2 = st.columns([5, 5])
    
    with r_col1:
        st.markdown("**Ranked Candidate List**")
        st.dataframe(
            ranked_results[['Tool', 'Suitability', 'deployment', 'barrier']].head(5).style.background_gradient(subset=['Suitability'], cmap="Greens"),
            use_container_width=True,
            hide_index=True  
        )
        
        if len(ranked_results) > 0:
            best_tool = ranked_results.iloc[0]
            st.success(f"**Top Recommendation:** {best_tool['Tool']} ({best_tool['Suitability']:.1f}%)")
        else:
            st.warning("No suitable tools found based on these preferences.")

    with r_col2:
        st.markdown("**Fit Analysis (Radar Chart)**")
        
        if len(ranked_results) > 0:
            top_tools = ranked_results.head(3)
            categories = ['Fidelity', 'Privacy', 'Flexibility', 'Ease of Use', 'Speed']
            
            fig = go.Figure()
            
            for idx, row in top_tools.iterrows():
                t_data = tools_df[tools_df['name'] == row['Tool']].iloc[0]
                
               
                r_vals = [
                    2 if t_data['hallucination_risk'] == 'High' else 10,
                    2 if t_data['deployment'] == 'Cloud' else 10, 
                    10 if t_data['layout_flex'] == 'Very High' else (6 if t_data['layout_flex']=='Medium' else 3),
                    2 if t_data['barrier'] == 'High' else (6 if t_data['barrier'] == 'Medium' else 10),
                    10 if t_data['throughput_mode'] == 'Batch' else 4
                ]
                r_vals += [r_vals[0]] 
                
                fig.add_trace(go.Scatterpolar(
                    r=r_vals, theta=categories + [categories[0]], fill='toself', name=row['Tool']
                ))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), margin=dict(t=30, b=30, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666; font-size: 12px;'>
        <p><b>GLAM ATR Decision Support Tool</b><br>
        This interactive tool accompanies the study on <i>"AI-Based Automated Text Recognition in Cultural Heritage Digitization: From Benchmarking to Decision Support for GLAM Institutions"</i>.<br>
        Note: The recommendations are based on heuristic logic and technical benchmarks defined in the study.</p>
    </div>
    """, 
    unsafe_allow_html=True
)