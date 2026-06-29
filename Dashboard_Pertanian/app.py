import streamlit as st
import pandas as pd
import joblib
import math
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Diagnostik Kesehatan Tanaman",
    layout="wide",
    page_icon=":material/eco:",
)

PALETTE = {
    "loam": "#2B2420",        
    "parchment": "#FBF6EA",   
    "husk": "#ECE1C0",        
    "moss": "#4B6043",        
    "terracotta": "#B85C38",  
    "slate": "#46524C",       
    "sky": "#4D6A7A",         
    "ochre": "#9C7A3C",       
}

st.markdown("""
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8a7c52;
        margin-bottom: 0.25rem;
    }

    .hero-wrap {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        border-bottom: 2px solid #2B2420;
        padding-bottom: 1.1rem;
        margin-bottom: 1.6rem;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .hero-title {
        font-family: 'Spectral', serif;
        font-weight: 600;
        font-size: 2.3rem;
        color: #2B2420;
        margin: 0;
        line-height: 1.15;
    }
    .hero-sub {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #5b5347;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        max-width: 540px;
    }
    .hero-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #46524C;
        text-align: right;
        line-height: 1.7;
        white-space: nowrap;
    }
    .hero-meta b { color: #2B2420; }

    .kpi-card {
        background: #ECE1C0;
        border-radius: 12px;
        padding: 0.9rem 1.05rem;
        border-left: 4px solid #4B6043;
        margin-bottom: 0.6rem;
    }
    .kpi-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #5b5347;
    }
    .kpi-value {
        font-family: 'Spectral', serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: #2B2420;
        margin-top: 0.15rem;
    }

    .ticket {
        border-radius: 16px;
        padding: 1.3rem 1.6rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        border: 1.5px dashed;
        margin-top: 0.4rem;
    }
    .ticket.healthy { background: #EBF0E5; border-color: #4B6043; }
    .ticket.unhealthy { background: #F5E6DA; border-color: #B85C38; }
    .ticket-status {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.06em;
        font-weight: 700;
        font-size: 1.05rem;
        text-transform: uppercase;
    }
    .ticket-status.healthy { color: #3d5236; }
    .ticket-status.unhealthy { color: #8e431f; }
    .ticket-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        color: #5b5347;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    .legend-row {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        color: #2B2420;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .legend-dot {
        width: 9px; height: 9px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    button[data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    [data-baseweb="tab-highlight"] { background-color: #4B6043 !important; }
    </style>
""", unsafe_allow_html=True)


def render_gauge_svg(value, color, size=116, stroke=10):
    """SVG gauge melingkar untuk menampilkan skor confidence."""
    value = max(0.0, min(100.0, value))
    r = (size - stroke) / 2
    circumference = 2 * math.pi * r
    offset = circumference * (1 - value / 100)
    c = size / 2
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#E1D4A8" stroke-width="{stroke}"></circle>
      <circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
        stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
        transform="rotate(-90 {c} {c})"></circle>
      <text x="{c}" y="{c + 7}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
        font-size="20" font-weight="700" fill="{color}">{value:.1f}%</text>
    </svg>
    """


def kpi_card_html(label, value, icon="insights", accent=PALETTE["moss"]):
    return (
        f'<div class="kpi-card" style="border-left-color:{accent};">'
        f'<div class="kpi-label">:material/{icon}: {label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )


def render_kpi_row(items):
    """items: list berisi tuple (label, value, icon, accent)"""
    cols = st.columns(len(items))
    for col, (label, value, icon, accent) in zip(cols, items):
        col.markdown(kpi_card_html(label, value, icon, accent), unsafe_allow_html=True)


def render_ticket(is_healthy, confidence, source_label="Input Tunggal"):
    status_class = "healthy" if is_healthy else "unhealthy"
    status_text = "Tanaman Sehat" if is_healthy else "Tanaman Tidak Sehat"
    icon = "verified" if is_healthy else "report"
    accent = PALETTE["moss"] if is_healthy else PALETTE["terracotta"]
    gauge = render_gauge_svg(confidence, accent)
    timestamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    html = f"""
    <div class="ticket {status_class}">
        <div>{gauge}</div>
        <div>
            <div class="ticket-status {status_class}">:material/{icon}: {status_text}</div>
            <div class="ticket-meta">
                SUMBER&nbsp;&nbsp;{source_label}<br>
                DIPROSES&nbsp;&nbsp;{timestamp}<br>
                MODEL&nbsp;&nbsp;Decision Tree &middot; 10 fitur (RFE)
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        "<span style=\"font-family:'Spectral',serif; font-weight:700; font-size:1.35rem; color:#2B2420;\">"
        ":material/eco: Smart Farming ML</span>",
        unsafe_allow_html=True,
    )
    st.caption("DASHBOARD DIAGNOSTIK AGRONOMI")
    st.divider()
    st.markdown(
        "Dashboard ini membaca **10 parameter** lingkungan, tanah, dan tanaman, "
        "lalu mengklasifikasikan status kesehatan tanaman menggunakan model "
        "**Decision Tree** yang sudah dioptimasi."
    )
    st.markdown("###### :material/category: Legenda kategori parameter")
    st.markdown(
        f'<div class="legend-row"><span class="legend-dot" style="background:{PALETTE["sky"]}"></span> Lingkungan</div>'
        f'<div class="legend-row"><span class="legend-dot" style="background:{PALETTE["ochre"]}"></span> Tanah</div>'
        f'<div class="legend-row"><span class="legend-dot" style="background:{PALETTE["moss"]}"></span> Tanaman</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("© 2026 · Decision Tree Classification · v2.0")

rfe_features = [
    'Elevation_Data', 'Canopy_Coverage', 'SAVI', 'Crop_Stress_Indicator',
    'Soil_pH', 'Organic_Matter', 'Soil_Moisture', 'Humidity',
    'Water_Flow', 'Pest_Damage'
]


@st.cache_resource  
def load_components():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False


model, scaler, is_loaded = load_components()

st.markdown(f"""
<div class="hero-wrap">
    <div>
        <div class="eyebrow">Laporan Diagnostik &middot; Precision Agriculture</div>
        <p class="hero-title">Sistem Prediksi Kesehatan Tanaman</p>
        <p class="hero-sub">Analisis multi-parameter lingkungan, tanah, dan tanaman secara
        <i>real-time</i> menggunakan model klasifikasi Decision Tree teroptimasi.</p>
    </div>
    <div class="hero-meta">
        MODEL&nbsp;&nbsp;<b>Decision Tree</b><br>
        FITUR&nbsp;&nbsp;<b>10 (hasil RFE)</b><br>
        STATUS&nbsp;&nbsp;<b>{"Model siap" if is_loaded else "Model belum dimuat"}</b>
    </div>
</div>
""", unsafe_allow_html=True)


def plot_evaluation(actuals, predictions):
    st.markdown("#### :material/query_stats: Evaluasi Model (Testing Data)")

    col_met, col_chart = st.columns([1.5, 2.5])

    with col_met:
        acc = accuracy_score(actuals, predictions)
        st.markdown(
            kpi_card_html("Akurasi prediksi keseluruhan", f"{acc*100:.2f}%", "insights", PALETTE["moss"]),
            unsafe_allow_html=True,
        )

        st.markdown("**Classification report:**")
        report_df = pd.DataFrame(classification_report(actuals, predictions, output_dict=True, labels=[0, 1], target_names=['Tidak Sehat (0)', 'Sehat (1)'], zero_division=0)).transpose()
        styled_report = report_df.style.format(precision=3).set_properties(
            **{'font-family': "'IBM Plex Mono', monospace"}
        )
        st.dataframe(styled_report, use_container_width=True)

    with col_chart:
        cm = confusion_matrix(actuals, predictions, labels=[0, 1])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Tidak Sehat', 'Sehat'],
            y=['Tidak Sehat', 'Sehat'],
            hoverongaps=False,
            colorscale=[[0, "#F6F1E4"], [1, "#6F8761"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": PALETTE["loam"]}
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["loam"]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_prediction_distribution(predictions, probas):
    st.markdown("#### :material/donut_large: Visualisasi Hasil Prediksi")

    sehat_count = int(sum(predictions))
    tidak_count = len(predictions) - sehat_count
    avg_conf = sum(max(p) * 100 for p in probas) / len(probas)

    render_kpi_row([
        ("Total sampel", f"{len(predictions)}", "table_chart", PALETTE["slate"]),
        ("Sehat", f"{sehat_count}", "verified", PALETTE["moss"]),
        ("Tidak sehat", f"{tidak_count}", "report", PALETTE["terracotta"]),
        ("Rata-rata confidence", f"{avg_conf:.1f}%", "speed", PALETTE["loam"]),
    ])

    col1, col2 = st.columns(2)

    with col1:
        pred_labels = ['Sehat' if p == 1 else 'Tidak Sehat' for p in predictions]
        df_pred = pd.DataFrame({'Status': pred_labels})
        fig1 = px.pie(df_pred, names='Status', hole=0.45,
                      color='Status',
                      color_discrete_map={'Sehat': PALETTE["moss"], 'Tidak Sehat': PALETTE["terracotta"]},
                      title='Distribusi Proporsi Kelas')
        fig1.update_layout(
            height=350, margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["loam"]),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        confidences = [max(p) * 100 for p in probas]
        df_conf = pd.DataFrame({'Confidence (%)': confidences})
        fig2 = px.histogram(df_conf, x='Confidence (%)', nbins=10,
                             title='Distribusi Confidence Score',
                             color_discrete_sequence=[PALETTE["slate"]],
                             marginal='box')
        fig2.update_layout(
            height=350, margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["loam"]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig2, use_container_width=True)


tab1, tab2, tab3 = st.tabs([
    ":material/tune: Input Tunggal",
    ":material/edit_note: Input Batch Manual",
    ":material/upload_file: Unggah Berkas Batch",
])

with tab1:
    st.markdown("#### Masukkan Parameter Observasi")

    col_env, col_soil, col_plant = st.columns(3)

    with col_env:
        with st.container(border=True):
            st.markdown("##### :material/water_drop: Parameter Lingkungan")
            elevation = st.number_input('Elevation (mdpl)', 0.0, 3000.0, 100.0)
            humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0)
            water_flow = st.slider('Water Flow (L/s)', 0.0, 100.0, 20.0)

    with col_soil:
        with st.container(border=True):
            st.markdown("##### :material/terrain: Kondisi Tanah")
            soil_ph = st.slider('Soil pH', 0.0, 14.0, 6.5)
            organic = st.slider('Organic Matter', 0.0, 100.0, 20.0)
            moisture = st.slider('Soil Moisture (%)', 0.0, 100.0, 25.0)

    with col_plant:
        with st.container(border=True):
            st.markdown("##### :material/grass: Indikator Tanaman")
            canopy = st.slider('Canopy Coverage (%)', 0.0, 200.0, 50.0)
            savi = st.slider('SAVI Index', -1.0, 1.0, 0.4)
            stress = st.slider('Crop Stress', 0.0, 100.0, 10.0)
            pest_damage = st.slider('Pest Damage', 0.0, 100.0, 5.0)

    single_input_df = pd.DataFrame({
        'Elevation_Data': [elevation], 'Canopy_Coverage': [canopy], 'SAVI': [savi],
        'Crop_Stress_Indicator': [stress], 'Soil_pH': [soil_ph], 'Organic_Matter': [organic],
        'Soil_Moisture': [moisture], 'Humidity': [humidity], 'Water_Flow': [water_flow],
        'Pest_Damage': [pest_damage]
    })

    st.markdown("---")
    col_btn, col_res = st.columns([1, 3])

    with col_btn:
        st.write("")  
        predict_btn = st.button("Proses Prediksi", type="primary", icon=":material/bolt:", use_container_width=True)

    with col_res:
        if predict_btn:
            if not is_loaded:
                st.error("File model/scaler belum siap! Pastikan 'model.pkl' dan 'scaler.pkl' ada di direktori yang sama.", icon=":material/error:")
            else:
                scaled = scaler.transform(single_input_df)
                pred = model.predict(scaled)[0]
                proba = model.predict_proba(scaled)[0]
                confidence = max(proba) * 100
                render_ticket(pred == 1, confidence, source_label="Input Tunggal")

with tab2:
    st.markdown("#### Input Multiple Baris Manual")
    st.write("Edit tabel di bawah ini secara langsung untuk memprediksi banyak skenario sekaligus.")

    empty_df = pd.DataFrame(columns=rfe_features)
    empty_df.loc[0] = [100.0, 50.0, 0.4, 10.0, 6.5, 20.0, 25.0, 60.0, 20.0, 5.0]

    edited_df = st.data_editor(empty_df, num_rows="dynamic", use_container_width=True)

    run_batch = st.button("Prediksi Data Tabel", type="primary", icon=":material/play_circle:")

    if run_batch:
        if not is_loaded:
            st.error("File model/scaler belum siap!", icon=":material/error:")
        elif edited_df.empty:
            st.warning("Tabel masih kosong! Tambahkan minimal satu baris observasi.", icon=":material/warning:")
        else:
            with st.spinner('Memproses data...'):
                data_to_predict = edited_df[rfe_features]
                scaled = scaler.transform(data_to_predict)
                preds = model.predict(scaled)
                probas = model.predict_proba(scaled)

                result_df = edited_df.copy()
                result_df.insert(0, 'STATUS PREDIKSI', ["Sehat" if p == 1 else "Tidak Sehat" for p in preds])
                result_df.insert(1, 'CONFIDENCE', [f"{max(p)*100:.2f}%" for p in probas])

                st.markdown("#### :material/table_chart: Hasil Klasifikasi")

                
                def highlight_status(val):
                    color = PALETTE["moss"] if val == 'Sehat' else PALETTE["terracotta"]
                    return f'background-color: {color}; color: #FBF6EA; font-weight: bold'

                st.dataframe(result_df.style.map(highlight_status, subset=['STATUS PREDIKSI']), use_container_width=True)
                plot_prediction_distribution(preds, probas)

with tab3:
    st.markdown("#### Prediksi Massal via File (CSV/Excel)")
    st.write("Sistem akan otomatis mengekstraksi metrik evaluasi jika terdapat kolom `Crop_Health_Label`.")

    uploaded_file = st.file_uploader("Seret dan lepas file dataset di sini", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

            with st.expander("Intip data asli", icon=":material/visibility:", expanded=False):
                st.dataframe(df_upload.head(), use_container_width=True)

            missing_cols = [col for col in rfe_features if col not in df_upload.columns]

            if missing_cols:
                st.error(f"Validasi gagal! Kolom berikut tidak ditemukan di file: `{', '.join(missing_cols)}`", icon=":material/error:")
            else:
                run_mass = st.button("Jalankan Prediksi Massal", type="primary", icon=":material/rocket_launch:", use_container_width=True)
                if run_mass:
                    if not is_loaded:
                        st.error("File model/scaler belum siap!", icon=":material/error:")
                    else:
                        with st.spinner('Menjalankan pipeline klasifikasi...'):
                            data_to_predict = df_upload[rfe_features]
                            scaled = scaler.transform(data_to_predict)
                            preds = model.predict(scaled)
                            probas = model.predict_proba(scaled)

                            df_result = df_upload.copy()
                            df_result.insert(0, 'PREDICTED_STATUS', ["Sehat" if p == 1 else "Tidak Sehat" for p in preds])
                            df_result.insert(1, 'CONFIDENCE', [f"{max(p)*100:.2f}%" for p in probas])

                            st.markdown("---")
                            if 'Crop_Health_Label' in df_upload.columns:
                                plot_evaluation(df_upload['Crop_Health_Label'], preds)
                            else:
                                st.info("Kolom `Crop_Health_Label` tidak ditemukan. Melewati evaluasi metrik.", icon=":material/info:")

                            plot_prediction_distribution(preds, probas)

                            st.markdown("#### :material/download: Download Hasil Analisis")
                            st.dataframe(df_result.head(10), use_container_width=True)

                            csv = df_result.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Unduh dataset lengkap (CSV)",
                                data=csv,
                                file_name='hasil_prediksi_batch.csv',
                                mime='text/csv',
                                type="primary",
                                icon=":material/download:",
                            )
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis saat membaca file: {e}", icon=":material/error:")
