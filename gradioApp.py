import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load model assets
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
le = joblib.load("label_encoder.pkl")

# -------------------------------
# Mapping functions
# -------------------------------

def map_inputs(data):
    return {
        "Gender": data["Gender"],
        "Age": data["Age"],
        "Height": data["Height"],
        "Weight": data["Weight"],
        "family_history_with_overweight": data["Family History"],
        "FAVC": data["High Calorie Food"],
        "FCVC": {"Rarely": 1, "Sometimes": 2, "Frequently": 3}[data["Vegetables"]],
        "NCP": data["Meals"],
        "CAEC": data["Snacking"],
        "SMOKE": data["Smoking"],
        "CH2O": data["Water"],
        "SCC": data["Calorie Monitoring"],
        "FAF": {"None": 0, "Light": 1, "Moderate": 2, "High": 3}[data["Activity"]],
        "TUE": {"Low": 0, "Medium": 1, "High": 2}[data["Screen Time"]],
        "CALC": data["Alcohol"],
        "MTRANS": data["Transport"]
    }

def clean_label(label):
    return label.replace("_", " ").replace("Type", "Type ")

def calculate_bmi(height, weight):
    if height > 0:
        bmi = weight / (height ** 2)
        return round(bmi, 1)
    return 0

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#3B82F6"
    elif bmi < 25:
        return "Normal", "#22C55E"
    elif bmi < 30:
        return "Overweight", "#F59E0B"
    else:
        return "Obese", "#EF4444"

# -------------------------------
# Chart Builder
# -------------------------------

CATEGORY_COLORS = {
    "Insufficient Weight": "#3B82F6",
    "Normal Weight":       "#22C55E",
    "Overweight Level I":  "#F59E0B",
    "Overweight Level II": "#F97316",
    "Obesity Type I":      "#EF4444",
    "Obesity Type II":     "#DC2626",
    "Obesity Type III":    "#991B1B",
}

def build_chart_html(prob_dict: dict) -> str:
    if not prob_dict:
        return ""

    bars_html = ""
    for label, prob in prob_dict.items():
        color = CATEGORY_COLORS.get(label, "#6B7280")
        pct   = prob * 100          # 0–100 for bar width
        pct_display = f"{pct:.1f}%"

        bars_html += f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        ">
            <!-- Label -->
            <div style="
                width: clamp(120px, 28%, 180px);
                flex-shrink: 0;
                color: #CBD5E1;
                font-size: clamp(0.7rem, 1.8vw, 0.82rem);
                text-align: right;
                line-height: 1.3;
            ">{label}</div>

            <!-- Bar track -->
            <div style="
                flex: 1;
                background: #1a1a2e;
                border-radius: 999px;
                height: 22px;
                overflow: hidden;
                min-width: 0;
            ">
                <div style="
                    width: {pct:.2f}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}cc, {color});
                    border-radius: 999px;
                    transition: width 0.6s cubic-bezier(.4,0,.2,1);
                    min-width: {('4px' if pct > 0 else '0')};
                "></div>
            </div>

            <!-- Percentage -->
            <div style="
                width: 44px;
                flex-shrink: 0;
                color: {color};
                font-size: clamp(0.72rem, 1.8vw, 0.85rem);
                font-weight: 700;
                text-align: right;
            ">{pct_display}</div>
        </div>
        """

    return f"""
    <div style="
        background: linear-gradient(145deg, #1e1e2e, #252535);
        border-radius: 14px;
        padding: clamp(14px, 3vw, 24px);
        border: 1px solid rgba(255,255,255,0.07);
        box-sizing: border-box;
        width: 100%;
    ">
        <p style="
            color: #A78BFA;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 0 0 16px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(167,139,250,0.15);
        ">How Confident Is the Model?</p>
        {bars_html}
        <p style="
            margin: 14px 0 0 0;
            color: #4B5563;
            font-size: 0.72rem;
            text-align: right;
        ">Each bar shows the model's confidence for that category</p>
    </div>
    """

# -------------------------------
# Prediction function
# -------------------------------

def predict(
    Gender, Age, Height, Weight,
    Family_History, High_Calorie_Food,
    Vegetables, Meals, Snacking,
    Smoking, Water, Calorie_Monitoring,
    Activity, Screen_Time, Alcohol, Transport
):
    data = {
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "Family History": Family_History,
        "High Calorie Food": High_Calorie_Food,
        "Vegetables": Vegetables,
        "Meals": Meals,
        "Snacking": Snacking,
        "Smoking": Smoking,
        "Water": Water,
        "Calorie Monitoring": Calorie_Monitoring,
        "Activity": Activity,
        "Screen Time": Screen_Time,
        "Alcohol": Alcohol,
        "Transport": Transport
    }

    mapped = map_inputs(data)
    df = pd.DataFrame([mapped])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    probs  = model.predict_proba(df)[0]
    classes = le.inverse_transform(np.arange(len(probs)))
    classes = [clean_label(c) for c in classes]

    best_idx   = np.argmax(probs)
    best_class = classes[best_idx]
    confidence = probs[best_idx] * 100

    # BMI
    bmi = calculate_bmi(Height, Weight)
    bmi_category, bmi_color = get_bmi_category(bmi)
    bmi_position = min(max((bmi - 10) / (45 - 10) * 100, 0), 100)

    severity_colors = {
        "Insufficient Weight": "#3B82F6",
        "Normal Weight":       "#22C55E",
        "Overweight Level I":  "#F59E0B",
        "Overweight Level II": "#F97316",
        "Obesity Type I":      "#EF4444",
        "Obesity Type II":     "#DC2626",
        "Obesity Type III":    "#991B1B",
    }
    color = severity_colors.get(best_class, "#6B7280")

    result_text = f"""
<div style="
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border-radius: 16px;
    padding: clamp(16px, 3vw, 28px);
    border-left: 5px solid {color};
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    font-family: 'Segoe UI', sans-serif;
    box-sizing: border-box;
    width: 100%;
">
    <!-- Prediction -->
    <div style="margin-bottom: 18px;">
        <p style="margin: 0 0 4px 0; color: #9CA3AF; font-size: 0.75rem;
                  text-transform: uppercase; letter-spacing: 1px;">
            Predicted Level
        </p>
        <h2 style="margin: 0; color: {color};
                   font-size: clamp(1.2rem, 4vw, 1.7rem);
                   font-weight: 700; line-height: 1.2;">
            🩺 {best_class}
        </h2>
    </div>

    <!-- Confidence -->
    <div style="margin-bottom: 18px;">
        <p style="color: #9CA3AF; font-size: 0.75rem; margin: 0 0 8px 0;
                  text-transform: uppercase; letter-spacing: 1px;">
            Confidence
        </p>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #374151; border-radius: 999px; height: 10px;
                        flex: 1; overflow: hidden; min-width: 0;">
                <div style="width: {confidence:.1f}%; height: 100%;
                            background: linear-gradient(90deg, {color}, {color}88);
                            border-radius: 999px;"></div>
            </div>
            <span style="color: white; font-weight: 700; font-size: 0.95rem;
                         white-space: nowrap;">{confidence:.1f}%</span>
        </div>
    </div>

    <hr style="border: none; border-top: 1px solid #2d2d45; margin: 14px 0;" />

    <!-- BMI -->
    <div>
        <p style="color: #9CA3AF; font-size: 0.75rem; margin: 0 0 10px 0;
                  text-transform: uppercase; letter-spacing: 1px;">
            Your BMI
        </p>
        <div style="background: #12121f; border-radius: 12px; padding: clamp(12px, 2.5vw, 16px);
                    border: 1px solid rgba(255,255,255,0.06);">

            <!-- Value + Badge -->
            <div style="display: flex; justify-content: space-between;
                        align-items: center; margin-bottom: 12px;
                        flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: clamp(1.7rem, 5vw, 2.3rem);
                                 font-weight: 800; color: {bmi_color};">{bmi}</span>
                    <span style="color: #6B7280; font-size: 0.8rem; margin-left: 4px;">kg/m²</span>
                </div>
                <span style="
                    background: {bmi_color}22; color: {bmi_color};
                    padding: 4px 14px; border-radius: 999px;
                    font-size: 0.8rem; font-weight: 600;
                    border: 1px solid {bmi_color}44; white-space: nowrap;
                ">{bmi_category}</span>
            </div>

            <!-- Scale bar -->
            <div style="position: relative;">
                <div style="
                    height: 8px; border-radius: 999px;
                    background: linear-gradient(90deg,
                        #3B82F6 0%, #22C55E 25%, #F59E0B 55%,
                        #EF4444 80%, #991B1B 100%);
                    margin-bottom: 6px; position: relative;
                ">
                    <div style="
                        position: absolute;
                        left: {bmi_position}%;
                        top: 50%;
                        transform: translate(-50%, -50%);
                        width: 15px; height: 15px;
                        background: white; border-radius: 50%;
                        border: 2px solid {bmi_color};
                        box-shadow: 0 0 6px {bmi_color};
                    "></div>
                </div>
                <div style="display: flex; justify-content: space-between;
                            color: #4B5563; font-size: 0.65rem;">
                    <span>10</span><span>18.5</span>
                    <span>25</span><span>30</span><span>45</span>
                </div>
            </div>

            <!-- Legend chips -->
            <div style="display: flex; gap: 5px; margin-top: 12px; flex-wrap: wrap;">
                <span style="font-size: 0.67rem; color: #3B82F6; background: #3B82F620;
                             padding: 2px 8px; border-radius: 4px;">&lt;18.5 Underweight</span>
                <span style="font-size: 0.67rem; color: #22C55E; background: #22C55E20;
                             padding: 2px 8px; border-radius: 4px;">18.5–24.9 Normal</span>
                <span style="font-size: 0.67rem; color: #F59E0B; background: #F59E0B20;
                             padding: 2px 8px; border-radius: 4px;">25–29.9 Overweight</span>
                <span style="font-size: 0.67rem; color: #EF4444; background: #EF444420;
                             padding: 2px 8px; border-radius: 4px;">30+ Obese</span>
            </div>
        </div>
    </div>

    <p style="margin: 16px 0 0 0; color: #6B7280; font-size: 0.73rem;
              font-style: italic; line-height: 1.5;">
        ⚠️ For informational use only. Please consult a healthcare professional.
    </p>
</div>
"""

    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return result_text, prob_dict


# -------------------------------
# Custom CSS
# -------------------------------

custom_css = """
*, *::before, *::after { box-sizing: border-box !important; }

body, .gradio-container {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    min-height: 100vh;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    overflow-x: hidden !important;
}

.gradio-container > * { max-width: 100% !important; }

.app-header {
    text-align: center;
    padding: 36px 16px 28px;
    background: linear-gradient(135deg, #1e1e3f, #2d2d5e);
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 16px 48px rgba(99,102,241,0.15);
}

.section-card {
    background: linear-gradient(145deg, #1e1e2e, #252535) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    margin-bottom: 16px !important;
}

label span, .gr-form label {
    color: #C4B5FD !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

input[type="number"],
.gr-dropdown select,
textarea,
.gr-input input {
    background: #2a2a3e !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: #E5E7EB !important;
    padding: 10px 14px !important;
    width: 100% !important;
}

input[type="number"]:focus,
.gr-dropdown select:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
    outline: none !important;
}

.gr-radio label {
    background: #2a2a3e !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    color: #D1D5DB !important;
    cursor: pointer !important;
    font-size: 0.85rem !important;
}

.gr-radio input:checked + label {
    background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
    border-color: #7C3AED !important;
    color: white !important;
}

.predict-btn button {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #2563EB 100%) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.predict-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(124,58,237,0.6) !important;
}

.section-title {
    color: #A78BFA !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 14px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(167,139,250,0.2) !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #1e1e2e; }
::-webkit-scrollbar-thumb { background: #4C1D95; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7C3AED; }

@media (max-width: 768px) {
    .app-header { padding: 22px 14px 18px; border-radius: 12px; margin-bottom: 14px; }
    .section-card { padding: 13px !important; }
    .main-row { flex-direction: column !important; }
    .main-row > div { width: 100% !important; min-width: 0 !important; }
    input[type="number"], select { font-size: 16px !important; }
}

@media (max-width: 480px) {
    .gradio-container { padding: 8px !important; }
    .section-card { padding: 11px !important; }
}
"""

# -------------------------------
# UI Layout
# -------------------------------

with gr.Blocks(css=custom_css, title="Obesity Level Predictor") as app:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <h1 style="
            margin: 0 0 10px 0;
            font-size: clamp(1.4rem, 5vw, 2.1rem);
            font-weight: 800;
            background: linear-gradient(135deg, #A78BFA, #60A5FA, #34D399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        ">🏥 Obesity Level Predictor</h1>
        <p style="
            margin: 0 auto;
            color: #9CA3AF;
            font-size: clamp(0.83rem, 2.5vw, 0.98rem);
            max-width: 480px;
            line-height: 1.6;
        ">
            Fill in your health and lifestyle details to get an AI-powered obesity level assessment.
        </p>
        <div style="display:flex; justify-content:center; gap:10px;
                    margin-top:16px; flex-wrap:wrap;">
            <span style="background:rgba(99,102,241,0.2); color:#A78BFA;
                         padding:5px 13px; border-radius:999px; font-size:0.75rem;
                         border:1px solid rgba(99,102,241,0.3);">ML-Powered</span>
            <span style="background:rgba(16,185,129,0.2); color:#34D399;
                         padding:5px 13px; border-radius:999px; font-size:0.75rem;
                         border:1px solid rgba(16,185,129,0.3);">⚡ Instant Results</span>
            <span style="background:rgba(59,130,246,0.2); color:#60A5FA;
                         padding:5px 13px; border-radius:999px; font-size:0.75rem;
                         border:1px solid rgba(59,130,246,0.3);">🔒 Privacy First</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False, elem_classes="main-row"):

        # ── LEFT COLUMN ──
        with gr.Column(scale=5, min_width=280):

            with gr.Group(elem_classes="section-card"):
                gr.HTML('<p class="section-title">Personal Information</p>')
                with gr.Row():
                    Gender = gr.Dropdown(["Male","Female"], label="Gender", value="Male")
                    Age    = gr.Number(label="Age (years)", value=25, minimum=1, maximum=120)
                with gr.Row():
                    Height = gr.Number(label="Height (m)",  value=1.70, minimum=0.5, maximum=2.5)
                    Weight = gr.Number(label="Weight (kg)", value=70,   minimum=10,  maximum=300)

            with gr.Group(elem_classes="section-card"):
                gr.HTML('<p class="section-title">🍽️ Eating Habits</p>')
                with gr.Row():
                    Family_History   = gr.Radio(["yes","no"], label="Family history of overweight?", value="no")
                    High_Calorie_Food = gr.Radio(["yes","no"], label="Frequent high-calorie food?",   value="no")
                with gr.Row():
                    Vegetables = gr.Dropdown(["Rarely","Sometimes","Frequently"],
                                             label="Vegetable consumption", value="Sometimes")
                    Snacking   = gr.Dropdown(["no","Sometimes","Frequently","Always"],
                                             label="Snacking between meals", value="Sometimes")
                with gr.Row():
                    Meals = gr.Number(label="Meals per day",      value=3,   minimum=1, maximum=10)
                    Water = gr.Number(label="Water intake (L/day)", value=2.0, minimum=0, maximum=10)
                Calorie_Monitoring = gr.Radio(["yes","no"], label="Do you monitor calories?", value="no")

            with gr.Group(elem_classes="section-card"):
                gr.HTML('<p class="section-title">🏃 Lifestyle & Habits</p>')
                with gr.Row():
                    Activity    = gr.Dropdown(["None","Light","Moderate","High"],
                                              label="Physical activity level", value="Light")
                    Screen_Time = gr.Dropdown(["Low","Medium","High"],
                                              label="Daily screen time", value="Medium")
                with gr.Row():
                    Smoking = gr.Radio(["yes","no"], label="Smoking", value="no")
                    Alcohol = gr.Dropdown(["no","Sometimes","Frequently","Always"],
                                          label="Alcohol consumption", value="Sometimes")

            with gr.Group(elem_classes="section-card"):
                gr.HTML('<p class="section-title">Transport</p>')
                Transport = gr.Dropdown(
                    ["Walking","Bike","Public_Transportation","Automobile"],
                    label="Primary mode of transport", value="Public_Transportation"
                )

            with gr.Row(elem_classes="predict-btn"):
                btn = gr.Button("Predict My Obesity Level", variant="primary")

        # ── RIGHT COLUMN ──
        with gr.Column(scale=4, min_width=280):

            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #1e1e2e, #252535);
                border-radius: 14px;
                padding: 15px 20px;
                border: 1px solid rgba(255,255,255,0.07);
                margin-bottom: 14px;
            ">
                <p style="color:#A78BFA; font-size:0.75rem; font-weight:700;
                           text-transform:uppercase; letter-spacing:1.5px;
                           margin:0 0 8px 0;">How to use</p>
                <ol style="color:#9CA3AF; font-size:0.84rem; line-height:1.85;
                            margin:0; padding-left:18px;">
                    <li>Fill in all fields on the left</li>
                    <li>Click <strong style="color:#C4B5FD;">Predict</strong></li>
                    <li>Review your result and BMI</li>
                    <li>Consult a doctor for medical advice</li>
                </ol>
            </div>
            """)

            output_text = gr.HTML(value="""
            <div style="
                background: linear-gradient(135deg, #1e1e2e, #252535);
                border-radius: 14px; padding: 36px 24px;
                border: 1px dashed rgba(167,139,250,0.3);
                text-align: center;
            ">
                <div style="font-size:2.4rem; margin-bottom:10px;">🩺</div>
                <p style="margin:0; font-size:0.93rem; color:#6B7280;">
                    Your prediction will appear here
                </p>
                <p style="margin:6px 0 0 0; font-size:0.78rem; color:#4B5563;">
                    Fill in the form and click Predict
                </p>
            </div>
            """)

            # Custom HTML chart replaces gr.BarPlot
            output_chart = gr.HTML(value="")

            gr.HTML("""
            <div style="
                margin-top: 14px;
                background: rgba(234,179,8,0.07);
                border: 1px solid rgba(234,179,8,0.2);
                border-radius: 10px;
                padding: 12px 16px;
                display: flex; gap: 10px; align-items: flex-start;
            ">
                <span style="font-size:1rem; flex-shrink:0;">⚠️</span>
                <p style="margin:0; color:#FCD34D; font-size:0.78rem; line-height:1.6;">
                    This tool is for <strong>informational purposes only</strong>
                    and does not constitute medical advice.
                    Always consult a qualified healthcare professional.
                </p>
            </div>
            """)

    # ── Event Handler ──
    def predict_and_format(
        Gender, Age, Height, Weight,
        Family_History, High_Calorie_Food,
        Vegetables, Meals, Snacking,
        Smoking, Water, Calorie_Monitoring,
        Activity, Screen_Time, Alcohol, Transport
    ):
        result_html, prob_dict = predict(
            Gender, Age, Height, Weight,
            Family_History, High_Calorie_Food,
            Vegetables, Meals, Snacking,
            Smoking, Water, Calorie_Monitoring,
            Activity, Screen_Time, Alcohol, Transport
        )
        chart_html = build_chart_html(prob_dict)
        return result_html, chart_html

    btn.click(
        fn=predict_and_format,
        inputs=[
            Gender, Age, Height, Weight,
            Family_History, High_Calorie_Food,
            Vegetables, Meals, Snacking,
            Smoking, Water, Calorie_Monitoring,
            Activity, Screen_Time, Alcohol, Transport
        ],
        outputs=[output_text, output_chart]
    )

app.launch(server_name="0.0.0.0", server_port=7860)