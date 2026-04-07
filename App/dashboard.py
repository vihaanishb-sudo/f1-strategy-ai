import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os
import json

st.set_page_config(
    page_title="F1 Race Strategy AI",
    page_icon="🏎️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; }
    h1, h2, h3 { color: #e10600 !important; }
    .section-card {
        background-color: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-left: 4px solid #e10600;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .section-desc {
        color: #999999;
        font-size: 0.92em;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .small-metric {
        background: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .small-metric-label {
        color: #888;
        font-size: 0.78em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .small-metric-value {
        color: #f0f0f0;
        font-size: 0.95em;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .fuel-bar-container {
        background: #2d2d2d;
        border-radius: 4px;
        height: 18px;
        width: 100%;
        margin: 6px 0;
        overflow: hidden;
    }
    .fuel-bar-optimal {
        background: linear-gradient(90deg, #e10600, #ff4444);
        height: 100%;
        border-radius: 4px;
        display: inline-block;
    }
    .stButton>button[kind="primary"] {
        background-color: #e10600 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5em 2em !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #ff1a1a !important;
    }
    .stButton>button { border-radius: 6px !important; }
    [data-testid="stMetricLabel"] { color: #999999 !important; }
    hr { border-color: #2d2d2d !important; }
    .stAlert { border-radius: 6px !important; }
    [data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 1px solid #2d2d2d;
    }
    thead tr th {
        background-color: #e10600 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────
import os

# This gets the directory where dashboard.py lives, then goes one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'Model', 'f1_strategy_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'Model', 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

@st.cache_resource
def load_circuit_strategies():
    with open(os.path.join(BASE_DIR, 'Data', 'circuit_strategies.json'), 'r') as f:
        return json.load(f)

@st.cache_resource
def load_degradation_rates():
    with open(os.path.join(BASE_DIR, 'Data', 'degradation_rates.json'), 'r') as f:
        return json.load(f)

@st.cache_resource
def load_sc_probability():
    with open(os.path.join(BASE_DIR, 'Data', 'sc_probability.json'), 'r') as f:
        return json.load(f)

@st.cache_resource
def load_winner_strategies():
    with open(os.path.join(BASE_DIR, 'Data', 'winner_strategies.json'), 'r') as f:
        return json.load(f)

@st.cache_resource
def load_fuel_loads():
    with open(os.path.join(BASE_DIR, 'Data', 'fuel_loads.json'), 'r') as f:
        return json.load(f)

model, feature_columns = load_model()
circuit_strategies    = load_circuit_strategies()
all_degradation_rates = load_degradation_rates()
sc_probability        = load_sc_probability()
winner_strategies     = load_winner_strategies()
fuel_loads            = load_fuel_loads()

# ── Constants ─────────────────────────────────────────────────────────────
compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2,
                'INTERMEDIATE': 3, 'WET': 4}

circuit_laps = {
    'Abu Dhabi Grand Prix':      58,
    'Australian Grand Prix':     58,
    'Austrian Grand Prix':       71,
    'Azerbaijan Grand Prix':     51,
    'Bahrain Grand Prix':        57,
    'Belgian Grand Prix':        44,
    'British Grand Prix':        52,
    'Canadian Grand Prix':       70,
    'Chinese Grand Prix':        56,
    'Dutch Grand Prix':          72,
    'Emilia Romagna Grand Prix': 63,
    'Hungarian Grand Prix':      70,
    'Italian Grand Prix':        53,
    'Japanese Grand Prix':       53,
    'Las Vegas Grand Prix':      50,
    'Mexico City Grand Prix':    71,
    'Miami Grand Prix':          57,
    'Monaco Grand Prix':         78,
    'Qatar Grand Prix':          57,
    'Saudi Arabian Grand Prix':  50,
    'Singapore Grand Prix':      62,
    'Spanish Grand Prix':        66,
    'São Paulo Grand Prix':      71,
    'United States Grand Prix':  56,
}

max_stint_laps = {
    'SOFT':   25,
    'MEDIUM': 35,
    'HARD':   50,
}

one_stop_only_circuits = [
    'Monaco Grand Prix',
    'Azerbaijan Grand Prix',
    'Saudi Arabian Grand Prix',
]

two_stop_only_circuits = [
    'Qatar Grand Prix',
]

def get_deg_rate(race_name, compound):
    global_fallback = {'SOFT': 0.04, 'MEDIUM': 0.025, 'HARD': 0.012}
    return all_degradation_rates.get(race_name, {}).get(
        compound, global_fallback.get(compound, 0.025))

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏎️ F1 Strategy AI")
st.sidebar.markdown("---")
st.sidebar.markdown("### Race Settings")
selected_race = st.sidebar.selectbox(
    "Select Circuit", sorted(list(circuit_laps.keys())))
default_laps = circuit_laps.get(selected_race, 57)
total_laps = st.sidebar.slider(
    "Total Race Laps", min_value=30, max_value=78, value=default_laps)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This App")
st.sidebar.markdown("""
<div style='color:#999; font-size:0.85em; line-height:1.8'>
Built using real F1 telemetry data from FastF1.<br>
<b style='color:#e10600'>Model:</b> Random Forest Regressor<br>
<b style='color:#e10600'>Data:</b> 2022–2024 seasons<br>
<b style='color:#e10600'>Races:</b> 46 Grand Prix events<br>
<b style='color:#e10600'>Max Error:</b> ±1.2s per lap<br>
<b style='color:#e10600'>Accuracy:</b> 99.1%
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 24px 0 8px 0'>
    <h1 style='font-size: 2.4em; margin-bottom: 4px'>🏎️ F1 Race Strategy AI</h1>
    <p style='color: #999; font-size: 1.05em; margin-top: 0'>
        Machine learning-powered race strategy prediction
        using real F1 telemetry data
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Helper functions ──────────────────────────────────────────────────────
def small_metric(label, value):
    return f"""
    <div class='small-metric'>
        <div class='small-metric-label'>{label}</div>
        <div class='small-metric-value'>{value}</div>
    </div>
    """

def simulate_strategy(strategy, model, compound_map, total_laps, race_name):
    total_time = 0
    lap = 1
    lap_times = []

    for compound, stint_laps in strategy:
        compound_code = compound_map.get(compound, 1)
        deg_rate = get_deg_rate(race_name, compound)

        for tyre_life in range(1, stint_laps + 1):
            row = {col: 0 for col in feature_columns}
            row['LapNumber']    = lap
            row['TyreLife']     = tyre_life
            row['CompoundCode'] = compound_code
            race_col = f'Race_{race_name}'
            if race_col in row:
                row[race_col] = 1

            input_df = pd.DataFrame([row])[feature_columns]
            pred = model.predict(input_df)[0]
            pred += deg_rate * tyre_life

            total_time += pred
            lap_times.append({'Lap': lap, 'LapTime': pred,
                              'Compound': compound, 'TyreLife': tyre_life})
            lap += 1

    total_time += (len(strategy) - 1) * 22
    return total_time, pd.DataFrame(lap_times)

def fast_stint_time(compound, stint_laps, start_lap, race_name):
    race_col      = f'Race_{race_name}'
    compound_code = compound_map.get(compound, 1)
    deg_rate      = get_deg_rate(race_name, compound)

    lap_nums   = np.arange(start_lap, start_lap + stint_laps)
    tyre_lives = np.arange(1, stint_laps + 1)

    rows      = np.zeros((stint_laps, len(feature_columns)))
    feat_list = list(feature_columns)

    rows[:, feat_list.index('LapNumber')]    = lap_nums
    rows[:, feat_list.index('TyreLife')]     = tyre_lives
    rows[:, feat_list.index('CompoundCode')] = compound_code
    if race_col in feat_list:
        rows[:, feat_list.index(race_col)] = 1

    preds = model.predict(rows)
    return float(np.sum(preds + deg_rate * tyre_lives))

def run_optimal_search(selected_race, total_laps):
    results = []
    hist             = circuit_strategies.get(selected_race, {})
    compounds_used   = hist.get('compounds_used', ['SOFT', 'MEDIUM', 'HARD'])
    search_compounds = compounds_used if compounds_used else ['SOFT', 'MEDIUM', 'HARD']
    force_one_stop   = selected_race in one_stop_only_circuits
    force_two_stop   = selected_race in two_stop_only_circuits

    if not force_two_stop:
        for c1 in search_compounds:
            for c2 in search_compounds:
                if c1 == c2:
                    continue
                for split in range(10, total_laps - 10):
                    if split > max_stint_laps.get(c1, 50):
                        continue
                    if (total_laps - split) > max_stint_laps.get(c2, 50):
                        continue
                    t1 = fast_stint_time(c1, split, 1, selected_race)
                    t2 = fast_stint_time(c2, total_laps - split,
                                         split + 1, selected_race)
                    results.append({
                        'Strategy': f'{c1} ({split}) → {c2} ({total_laps - split})',
                        'Stops': 1, 'Time': t1 + t2 + 22,
                        'Laps': [(c1, split), (c2, total_laps - split)]
                    })

    if not force_one_stop:
        for c1 in search_compounds:
            for c2 in search_compounds:
                for c3 in search_compounds:
                    if len(set([c1, c2, c3])) < 2:
                        continue
                    for s1 in range(8, total_laps - 16, 3):
                        if s1 > max_stint_laps.get(c1, 50):
                            continue
                        for s2 in range(8, total_laps - s1 - 8, 3):
                            if s2 > max_stint_laps.get(c2, 50):
                                continue
                            s3 = total_laps - s1 - s2
                            if s3 < 8 or s3 > max_stint_laps.get(c3, 50):
                                continue
                            t1 = fast_stint_time(c1, s1, 1, selected_race)
                            t2 = fast_stint_time(c2, s2, s1 + 1, selected_race)
                            t3 = fast_stint_time(c3, s3, s1 + s2 + 1, selected_race)
                            results.append({
                                'Strategy': f'{c1} ({s1}) → {c2} ({s2}) → {c3} ({s3})',
                                'Stops': 2, 'Time': t1 + t2 + t3 + 44,
                                'Laps': [(c1, s1), (c2, s2), (c3, s3)]
                            })

    results_df = pd.DataFrame(results)
    best_row   = results_df.loc[results_df['Time'].idxmin()]
    return best_row, results_df.nsmallest(5, 'Time')

def make_lap_chart(lap_df, title):
    fig    = go.Figure()
    colors = {'SOFT': '#e10600', 'MEDIUM': '#ffd700', 'HARD': '#f0f0f0'}
    for compound in lap_df['Compound'].unique():
        d = lap_df[lap_df['Compound'] == compound]
        fig.add_trace(go.Scatter(
            x=d['Lap'], y=d['LapTime'],
            mode='lines+markers', name=compound,
            line=dict(color=colors.get(compound, '#888'), width=2),
            marker=dict(size=4)
        ))
    fig.update_layout(
        title=title,
        xaxis_title='Lap Number',
        yaxis_title='Lap Time (seconds)',
        template='plotly_dark', height=400,
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        font=dict(color='#cccccc'),
        title_font=dict(color='#e10600'),
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════
# SAFETY CAR BANNER
# ════════════════════════════════════════════════════════════════════════════
sc_info = sc_probability.get(selected_race, {})
if sc_info:
    sc_prob     = sc_info.get('sc_probability', 0)
    hist_prob   = sc_info.get('historical_prob', 0)
    track_score = sc_info.get('track_score', 0)
    risk_level  = sc_info.get('risk_level', 'UNKNOWN')
    avg_sc_laps = sc_info.get('avg_sc_laps', 0)
    races_used  = sc_info.get('races_analysed', 0)
    sc_occ      = sc_info.get('sc_occurrences', 0)

    sc_bar_filled = '█' * int(sc_prob * 20)
    sc_bar_empty  = '░' * (20 - int(sc_prob * 20))

    color_map = {
        'VERY HIGH': '#ff4444',
        'HIGH':      '#ff8c00',
        'MEDIUM':    '#ffd700',
        'LOW':       '#44bb44',
    }
    risk_color = color_map.get(risk_level, '#888')

    strategy_tip = {
        'VERY HIGH': 'Consider a flexible 2-stop strategy. A SC period could trigger a free pit stop — be ready to react.',
        'HIGH':      'A SC is more likely than not. Build flexibility into your strategy and stay alert for pit window opportunities.',
        'MEDIUM':    'SC is possible. A standard strategy should work, but be prepared to adapt if conditions change.',
        'LOW':       'SC is unlikely. Commit to your pre-planned strategy with confidence.',
    }.get(risk_level, '')

    st.markdown(f"""
    <div style='background:#1a1a1a; border:1px solid #2d2d2d;
                border-left:4px solid {risk_color};
                border-radius:6px; padding:16px 20px; margin-bottom:16px'>
        <div style='display:flex; justify-content:space-between; align-items:center'>
            <div>
                <span style='font-size:1.1em; font-weight:bold; color:{risk_color}'>
                    🚦 Safety Car Risk: {risk_level}
                </span>
                <span style='color:#888; font-size:0.85em; margin-left:12px'>
                    Based on {races_used} races ({sc_occ} with SC/VSC)
                </span>
            </div>
            <div style='font-size:1.4em; font-weight:bold; color:{risk_color}'>
                {sc_prob:.0%}
            </div>
        </div>
        <div style='margin:8px 0; font-family:monospace;
                    color:{risk_color}; letter-spacing:1px'>
            {sc_bar_filled}<span style='color:#333'>{sc_bar_empty}</span>
        </div>
        <div style='display:flex; gap:24px; margin:8px 0;
                    font-size:0.85em; color:#aaa'>
            <span>📊 Historical rate: <b style='color:#ccc'>{hist_prob:.0%}</b></span>
            <span>🏁 Track characteristics: <b style='color:#ccc'>{track_score:.0%}</b></span>
            <span>⏱ Avg SC laps when deployed: <b style='color:#ccc'>{avg_sc_laps:.0f}</b></span>
        </div>
        <div style='margin-top:8px; color:#bbb; font-size:0.88em'>
            💡 <i>{strategy_tip}</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REAL VS PREDICTED
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-card'>
    <h3 style='margin:0 0 4px 0'>📊 2024 Race Winner — Actual Strategy</h3>
    <p class='section-desc'>
        This section shows what the 2024 race winner actually did at this circuit,
        then runs the AI's recommendation alongside it for comparison. This validates
        how closely the model matches real-world F1 strategy decisions. Small differences
        are expected — real races involve safety cars, traffic and live reactions that
        no model can fully predict.
    </p>
</div>
""", unsafe_allow_html=True)

winner_data = winner_strategies.get(selected_race)

if not winner_data:
    st.info("ℹ️ No 2024 winner data available for this circuit.")
else:
    winner_name   = winner_data['winner_full']
    winner_abbr   = winner_data['winner']
    real_strategy = winner_data['strategy']
    real_pits     = winner_data['pit_stops']
    real_stints   = [(s['compound'], s['laps']) for s in real_strategy]
    real_label    = ' → '.join(
        [f"{s['compound']} ({s['laps']})" for s in real_strategy])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(small_metric("🏆 2024 Winner", winner_name),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(small_metric("🛞 Actual Strategy", real_label),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(small_metric("🔧 Pit Stops", str(real_pits)),
                    unsafe_allow_html=True)
    with col4:
        hist_stops = circuit_strategies.get(selected_race, {}).get(
            'typical_stops', '—')
        st.markdown(small_metric("📈 Typical Stops (2024)", str(hist_stops)),
                    unsafe_allow_html=True)

    if st.button("🔍 Run AI & Compare to 2024 Reality"):
        with st.spinner("🔄 Running AI strategy search..."):
            best_row, _ = run_optimal_search(selected_race, total_laps)

        ai_strategy = best_row['Laps']
        ai_label    = best_row['Strategy']

        real_sim_time, real_lap_df = simulate_strategy(
            real_stints, model, compound_map, total_laps, selected_race)
        ai_sim_time, ai_lap_df = simulate_strategy(
            ai_strategy, model, compound_map, total_laps, selected_race)

        diff = ai_sim_time - real_sim_time

        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏆 {winner_abbr} Strategy (simulated)",
                  f"{int(real_sim_time//60)}m {real_sim_time%60:.1f}s")
        c2.metric("🤖 AI Recommended (simulated)",
                  f"{int(ai_sim_time//60)}m {ai_sim_time%60:.1f}s",
                  delta=f"{diff:+.1f}s vs actual")
        c3.metric("📐 Gap", f"{abs(diff):.1f}s",
                  delta="AI faster" if diff < 0 else f"{winner_abbr} faster")

        st.markdown(f"**🤖 AI:** `{ai_label}`")
        st.markdown(f"**🏆 {winner_abbr}:** `{real_label}`")

        if abs(diff) < 5:
            st.success(
                f"✅ Strong validation — AI is within **{abs(diff):.1f}s** "
                f"of what the 2024 winner actually executed.")
        elif diff < 0:
            st.success(
                f"🤖 AI found a theoretically faster strategy by "
                f"**{abs(diff):.1f}s** under perfect conditions.")
        else:
            st.info(
                f"🏆 {winner_abbr}'s strategy was **{abs(diff):.1f}s** faster "
                f"in simulation — real-time race factors like SC timing and "
                f"traffic give experienced teams an edge the model cannot see.")

        fig = go.Figure()
        colors = {'SOFT': '#e10600', 'MEDIUM': '#ffd700', 'HARD': '#f0f0f0'}

        for compound in real_lap_df['Compound'].unique():
            d = real_lap_df[real_lap_df['Compound'] == compound]
            fig.add_trace(go.Scatter(
                x=d['Lap'], y=d['LapTime'], mode='lines+markers',
                name=f'{winner_abbr} Actual — {compound}',
                line=dict(color=colors.get(compound, '#888'),
                         width=2, dash='solid'),
                marker=dict(size=4)
            ))
        for compound in ai_lap_df['Compound'].unique():
            d = ai_lap_df[ai_lap_df['Compound'] == compound]
            fig.add_trace(go.Scatter(
                x=d['Lap'], y=d['LapTime'], mode='lines+markers',
                name=f'AI — {compound}',
                line=dict(color=colors.get(compound, '#888'),
                         width=2, dash='dash'),
                marker=dict(size=4, symbol='diamond')
            ))
        fig.update_layout(
            title=f'{winner_abbr} Actual vs AI — {selected_race} 2024',
            xaxis_title='Lap Number', yaxis_title='Lap Time (seconds)',
            template='plotly_dark', height=420,
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font=dict(color='#cccccc'),
            title_font=dict(color='#e10600'),
            legend=dict(orientation='h', yanchor='bottom',
                       y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Stint Breakdown")
        breakdown = []
        for i, s in enumerate(real_strategy):
            breakdown.append({
                'Stint': i + 1, 'Source': f'🏆 {winner_abbr}',
                'Compound': s['compound'], 'Laps': s['laps'],
                'Avg Lap': f"{s['avg_lap']:.2f}s",
                'Best Lap': f"{s['fastest_lap']:.2f}s",
            })
        for i, (c, l) in enumerate(ai_strategy):
            breakdown.append({
                'Stint': i + 1, 'Source': '🤖 AI',
                'Compound': c, 'Laps': l, 'Avg Lap': '—', 'Best Lap': '—',
            })
        st.dataframe(pd.DataFrame(breakdown), use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AI OPTIMAL STRATEGY FINDER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-card'>
    <h3 style='margin:0 0 4px 0'>🤖 AI Optimal Strategy Finder</h3>
    <p class='section-desc'>
        The AI exhaustively tests every valid 1-stop and 2-stop strategy for this
        circuit using circuit-specific tyre degradation rates derived from real
        FastF1 telemetry. It returns the fastest predicted strategy and the top 5
        alternatives so you can see how much time separates different approaches.
        Strategies are filtered to respect F1 regulations — at least two different
        compounds must be used, and stint lengths reflect real-world tyre windows.
    </p>
</div>
""", unsafe_allow_html=True)

hist       = circuit_strategies.get(selected_race, {})
comp_used  = hist.get('compounds_used', ['SOFT', 'MEDIUM', 'HARD'])
avg_stints = hist.get('avg_stint_lengths', {})

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(small_metric("🔴 Soft Degradation",
        f"{get_deg_rate(selected_race, 'SOFT'):.4f}s/lap"),
        unsafe_allow_html=True)
with col2:
    st.markdown(small_metric("🟡 Medium Degradation",
        f"{get_deg_rate(selected_race, 'MEDIUM'):.4f}s/lap"),
        unsafe_allow_html=True)
with col3:
    st.markdown(small_metric("⚪ Hard Degradation",
        f"{get_deg_rate(selected_race, 'HARD'):.4f}s/lap"),
        unsafe_allow_html=True)

if avg_stints:
    with st.expander("📐 Historical avg stint lengths (top 5, 2024)"):
        for c, v in avg_stints.items():
            st.markdown(f"- **{c}:** {v:.0f} laps")

if st.button("🔍 Find Optimal Strategy", type="primary"):
    with st.spinner("🔄 Testing all strategy combinations..."):
        status = st.empty()
        status.info("⏳ Searching 1-stop and 2-stop combinations...")
        best_row, top5 = run_optimal_search(selected_race, total_laps)
        status.empty()

    best_time     = best_row['Time']
    best_strategy = best_row['Laps']

    st.session_state['best_strategy']       = best_strategy
    st.session_state['best_strategy_label'] = best_row['Strategy']
    st.session_state['best_strategy_race']  = selected_race

    mins = int(best_time // 60)
    secs = best_time % 60

    st.markdown(f"""
    <div style='background:#1a1a1a; border:1px solid #2d2d2d;
                border-left:4px solid #e10600; border-radius:6px;
                padding:16px 20px; margin:12px 0'>
        <div style='font-size:1.3em; font-weight:bold; color:#e10600'>
            🏆 Optimal Strategy: {best_row['Strategy']}
        </div>
        <div style='font-size:1.05em; color:#ccc; margin-top:4px'>
            ⏱️ Predicted Race Time: {mins}m {secs:.2f}s
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Top 5 Strategies")
    for idx, (_, row) in enumerate(top5.iterrows()):
        t    = row['Time']
        diff = t - best_time
        badge = "🥇" if idx == 0 else f"#{idx+1}"
        st.markdown(
            f"{badge} **{row['Strategy']}** ({row['Stops']}-stop) "
            f"— {int(t//60)}m {t%60:.1f}s "
            f"{'(fastest)' if idx == 0 else f'(+{diff:.1f}s)'}")

    _, lap_df = simulate_strategy(
        best_strategy, model, compound_map, total_laps, selected_race)
    st.plotly_chart(
        make_lap_chart(lap_df, f'Optimal Strategy — {selected_race}'),
        use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# FUEL LOAD CALCULATOR — directly under optimal strategy
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-card' style='margin-top:16px'>
    <h3 style='margin:0 0 4px 0'>⛽ Optimal Fuel Load</h3>
    <p class='section-desc'>
        Every kilogram of unnecessary fuel costs approximately 0.032 seconds
        per lap. Real F1 teams calculate the minimum fuel load required to
        finish the race, adding only a small safety buffer. This section shows
        the optimal fuel load for this circuit, how much time is lost by
        carrying a full 110kg tank instead, and the per-lap cost of excess fuel.
    </p>
</div>
""", unsafe_allow_html=True)

fuel_data = fuel_loads.get(selected_race)

if fuel_data:
    optimal_kg      = fuel_data['optimal_load_kg']
    full_kg         = fuel_data['full_load_kg']
    excess_kg       = fuel_data['excess_fuel_kg']
    time_penalty    = fuel_data['time_penalty_total_s']
    avg_lap_penalty = fuel_data['avg_lap_penalty_s']
    burn_rate       = fuel_data['burn_rate_per_lap']
    fuel_needed     = fuel_data['fuel_needed_kg']
    safety_margin   = fuel_data['safety_margin_kg']

    # Fuel bar visualisation
    bar_pct = int((optimal_kg / full_kg) * 100)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(small_metric("⛽ Optimal Fuel Load",
            f"{optimal_kg} kg"), unsafe_allow_html=True)
    with col2:
        st.markdown(small_metric("🔥 Fuel Burn Rate",
            f"{burn_rate} kg/lap"), unsafe_allow_html=True)
    with col3:
        st.markdown(small_metric("⚖️ Excess vs Full Tank",
            f"{excess_kg} kg saved"), unsafe_allow_html=True)
    with col4:
        st.markdown(small_metric("⏱️ Total Time Saved",
            f"{time_penalty:.1f}s vs 110kg"), unsafe_allow_html=True)

    # Visual fuel bar
    st.markdown(f"""
    <div style='margin: 16px 0 8px 0'>
        <div style='display:flex; justify-content:space-between;
                    font-size:0.85em; color:#888; margin-bottom:4px'>
            <span>0 kg</span>
            <span style='color:#e10600; font-weight:bold'>
                Optimal: {optimal_kg}kg ({bar_pct}% of max)
            </span>
            <span>110 kg (max)</span>
        </div>
        <div style='background:#2d2d2d; border-radius:6px;
                    height:22px; width:100%; position:relative'>
            <div style='background: linear-gradient(90deg, #e10600, #ff6644);
                        width:{bar_pct}%; height:100%;
                        border-radius:6px; position:absolute; top:0; left:0'>
            </div>
            <div style='position:absolute; top:0; left:0; width:100%;
                        height:100%; display:flex; align-items:center;
                        justify-content:center; font-size:0.78em;
                        font-weight:bold; color:white; z-index:2'>
                {optimal_kg}kg optimal &nbsp;|&nbsp; {excess_kg}kg saved
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown
    with st.expander("📋 Full fuel load breakdown"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Race laps | {fuel_data['race_laps']} |
        | Fuel burn rate | {burn_rate} kg/lap |
        | Minimum fuel needed | {fuel_needed} kg |
        | Safety buffer | {safety_margin} kg |
        | **Optimal load** | **{optimal_kg} kg** |
        | Maximum allowed | {full_kg} kg |
        | Excess if full tank | {excess_kg} kg |
        | Time cost per lap (excess) | {avg_lap_penalty:.4f}s/lap |
        | **Total time penalty (full tank)** | **{time_penalty:.1f}s** |
        """)

        st.markdown(f"""
        > 💡 **What this means:** Starting on {optimal_kg}kg instead of a full
        110kg tank saves **{time_penalty:.1f} seconds** across the entire race
        — equivalent to roughly {time_penalty/22:.1f} pit stop(s) worth of time.
        The {excess_kg}kg of unnecessary fuel adds {avg_lap_penalty:.4f}s
        to every single lap.
        """)

else:
    st.info("ℹ️ No fuel load data available for this circuit.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CUSTOM STRATEGY SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-card'>
    <h3 style='margin:0 0 4px 0'>⚙️ Custom Strategy Simulator</h3>
    <p class='section-desc'>
        Build your own race strategy manually by choosing your compounds and
        how many laps you want to run on each. The model predicts your total
        race time using circuit-specific lap times and tyre degradation rates.
        Use this to test unconventional strategies or validate your own thinking
        against the AI's recommendation above.
    </p>
</div>
""", unsafe_allow_html=True)

num_stints = st.radio("Number of Stints (pit stops + 1)", [2, 3])
stints     = []
cols       = st.columns(num_stints)

for i, col in enumerate(cols):
    with col:
        st.markdown(f"**Stint {i+1}**")
        compound = st.selectbox("Tyre Compound", ['SOFT', 'MEDIUM', 'HARD'],
                                key=f"compound_{i}")
        laps     = st.number_input(
            f"Laps (max {max_stint_laps.get(compound, 50)})",
            min_value=5,
            max_value=max_stint_laps.get(compound, 50),
            value=min(20, max_stint_laps.get(compound, 50)),
            key=f"laps_{i}")
        stints.append((compound, laps))

if st.button("▶️ Simulate My Strategy"):
    total_time, lap_df = simulate_strategy(
        stints, model, compound_map, total_laps, selected_race)

    mins = int(total_time // 60)
    secs = total_time % 60

    st.markdown(f"""
    <div style='background:#1a1a1a; border:1px solid #2d2d2d;
                border-left:4px solid #e10600; border-radius:6px;
                padding:16px 20px; margin:12px 0'>
        <div style='font-size:1.05em; color:#ccc'>
            ⏱️ Predicted Race Time:
            <b style='color:#e10600'>{mins}m {secs:.2f}s</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        make_lap_chart(lap_df, f'Custom Strategy — {selected_race}'),
        use_container_width=True)

    with st.expander("📋 View lap-by-lap data"):
        st.dataframe(lap_df, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AI VS YOUR STRATEGY
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-card'>
    <h3 style='margin:0 0 4px 0'>⚔️ AI vs Your Strategy</h3>
    <p class='section-desc'>
        Run the AI Optimal Strategy Finder first, then enter your own strategy
        here to compare head-to-head. Both are simulated through the same model
        so the comparison is fully fair. The gap in predicted time shows exactly
        how much your strategy gains or loses versus the AI's recommendation.
    </p>
</div>
""", unsafe_allow_html=True)

if 'best_strategy' not in st.session_state:
    st.warning(
        "⚠️ Run the **AI Optimal Strategy Finder** first to generate "
        "a strategy to compare against.")
else:
    st.markdown(
        f"**🤖 AI Optimal:** `{st.session_state['best_strategy_label']}`")

    num_stints_compare = st.radio("Number of Stints", [2, 3],
                                   key="compare_stints")
    compare_stints = []
    cols = st.columns(num_stints_compare)

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Stint {i+1}**")
            compound = st.selectbox("Tyre Compound", ['SOFT', 'MEDIUM', 'HARD'],
                                    key=f"compare_compound_{i}")
            laps = st.number_input("Laps on this tyre",
                                   min_value=5, max_value=60,
                                   value=20, key=f"compare_laps_{i}")
            compare_stints.append((compound, laps))

    if st.button("⚔️ Compare Strategies"):
        ai_time, ai_lap_df = simulate_strategy(
            st.session_state['best_strategy'],
            model, compound_map, total_laps,
            st.session_state['best_strategy_race'])

        custom_time, custom_lap_df = simulate_strategy(
            compare_stints, model, compound_map, total_laps, selected_race)

        diff   = abs(ai_time - custom_time)
        winner = "🤖 AI Strategy" if ai_time < custom_time else "👤 Your Strategy"

        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 AI Strategy",
                  f"{int(ai_time//60)}m {ai_time%60:.1f}s")
        c2.metric("👤 Your Strategy",
                  f"{int(custom_time//60)}m {custom_time%60:.1f}s",
                  delta=f"+{diff:.1f}s" if custom_time > ai_time
                  else f"-{diff:.1f}s")
        c3.metric("🏆 Winner", winner, delta=f"{diff:.1f}s faster")

        fig = go.Figure()
        colors = {'SOFT': '#e10600', 'MEDIUM': '#ffd700', 'HARD': '#f0f0f0'}

        for compound in ai_lap_df['Compound'].unique():
            d = ai_lap_df[ai_lap_df['Compound'] == compound]
            fig.add_trace(go.Scatter(
                x=d['Lap'], y=d['LapTime'], mode='lines+markers',
                name=f'AI — {compound}',
                line=dict(color=colors.get(compound, '#888'),
                         width=2, dash='solid'),
                marker=dict(size=4)
            ))
        for compound in custom_lap_df['Compound'].unique():
            d = custom_lap_df[custom_lap_df['Compound'] == compound]
            fig.add_trace(go.Scatter(
                x=d['Lap'], y=d['LapTime'], mode='lines+markers',
                name=f'You — {compound}',
                line=dict(color=colors.get(compound, '#888'),
                         width=2, dash='dash'),
                marker=dict(size=4, symbol='diamond')
            ))

        fig.update_layout(
            title=f'AI vs Your Strategy — {selected_race}',
            xaxis_title='Lap Number', yaxis_title='Lap Time (seconds)',
            template='plotly_dark', height=420,
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font=dict(color='#cccccc'),
            title_font=dict(color='#e10600'),
            legend=dict(orientation='h', yanchor='bottom',
                       y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        if ai_time < custom_time:
            st.success(
                f"🤖 The AI strategy is **{diff:.1f}s faster** "
                f"than your strategy.")
        else:
            st.success(
                f"👤 Your strategy beats the AI by **{diff:.1f}s!** Impressive.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — UNDERCUT / OVERCUT ANALYSER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='section-card'>
    <h3 style='margin:0 0 4px 0'>🔄 Undercut / Overcut Analyser</h3>
    <p class='section-desc'>
        Use this mid-race tool to decide whether to pit now or stay out.
        An <b>undercut</b> means pitting first, getting clean air on fresh tyres,
        and posting faster laps before your rival pits — hoping to emerge ahead.
        An <b>overcut</b> means staying out longer while your rival pits,
        banking track position and trusting your pace holds.
        Enter your current lap, tyre age and gap to the car ahead below.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    current_lap  = st.number_input("Current Lap", min_value=1,
                                    max_value=total_laps, value=20)
with col2:
    current_tyre = st.selectbox("Current Tyre", ['SOFT', 'MEDIUM', 'HARD'],
                                 key="uc_tyre")
with col3:
    tyre_age     = st.number_input("Tyre Age (laps)", min_value=1,
                                    max_value=50, value=10)
with col4:
    gap_ahead    = st.number_input("Gap to Car Ahead (s)",
                                    min_value=0.0, max_value=60.0,
                                    value=2.0, step=0.1)

next_tyre = st.selectbox("Tyre to fit if you pit now",
                          ['SOFT', 'MEDIUM', 'HARD'], key="uc_next_tyre")

if st.button("🔄 Analyse Undercut / Overcut"):
    remaining_laps   = total_laps - current_lap
    pit_stop_loss    = 22
    deg_rate_current = get_deg_rate(selected_race, current_tyre)
    deg_rate_new     = get_deg_rate(selected_race, next_tyre)

    undercut_time  = fast_stint_time(
        next_tyre, remaining_laps, current_lap + 1, selected_race)
    undercut_total = undercut_time + pit_stop_loss

    overcut_extra  = 3
    overcut_stay   = fast_stint_time(
        current_tyre, overcut_extra, current_lap + 1, selected_race)
    overcut_new    = fast_stint_time(
        next_tyre, remaining_laps - overcut_extra,
        current_lap + overcut_extra + 1, selected_race)
    overcut_total  = overcut_stay + overcut_new + pit_stop_loss

    current_penalty = deg_rate_current * tyre_age
    laps_to_recover = pit_stop_loss / current_penalty if current_penalty > 0 else 999

    undercut_viable = (gap_ahead < pit_stop_loss and
                       laps_to_recover < remaining_laps)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#1a1a1a; border:1px solid #2d2d2d;
                    border-left:4px solid #4488ff; border-radius:6px;
                    padding:14px; margin-bottom:8px'>
            <b style='color:#4488ff; font-size:1.05em'>
                🔵 UNDERCUT — Pit Now
            </b>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Predicted time (rest of race)",
                  f"{int(undercut_total//60)}m {undercut_total%60:.1f}s")
        if undercut_viable:
            st.success(
                f"✅ Undercut looks viable. Gap of {gap_ahead}s is within "
                f"the pit window. Need ~{laps_to_recover:.1f} laps to recover "
                f"pit stop time — {remaining_laps} remaining.")
        else:
            st.error(
                f"❌ Undercut risky. Gap of {gap_ahead}s may not be enough. "
                f"Need ~{laps_to_recover:.1f} laps to recover, "
                f"only {remaining_laps} remaining.")

    with c2:
        st.markdown("""
        <div style='background:#1a1a1a; border:1px solid #2d2d2d;
                    border-left:4px solid #ff8c00; border-radius:6px;
                    padding:14px; margin-bottom:8px'>
            <b style='color:#ff8c00; font-size:1.05em'>
                🟠 OVERCUT — Stay Out 3 More Laps
            </b>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Predicted time (rest of race)",
                  f"{int(overcut_total//60)}m {overcut_total%60:.1f}s")
        st.info(
            f"Stay out for 3 more laps, then pit. "
            f"Works best when gap ahead is large or a SC is likely.")

    st.markdown("---")
    if undercut_viable:
        st.success(
            f"### 🏆 Recommendation: UNDERCUT — "
            f"Gap of {gap_ahead}s is within the pit window. "
            f"Pit now to gain track position on fresh tyres.")
    elif gap_ahead > 25:
        st.info(
            f"### 💡 Recommendation: STAY OUT — "
            f"Gap of {gap_ahead}s is too large to undercut. "
            f"Focus on managing tyre life and wait for a safety car opportunity.")
    else:
        st.warning(
            f"### ⚠️ Recommendation: STAY OUT FOR NOW — "
            f"Gap of {gap_ahead}s is borderline. Stay out until the gap drops "
            f"below {pit_stop_loss:.0f}s or a safety car is deployed.")

    st.markdown(
        f"📊 **Degradation context:** Your {current_tyre} at age {tyre_age} "
        f"is adding **{deg_rate_current * tyre_age:.2f}s** of penalty per lap. "
        f"A fresh {next_tyre} starts at **{deg_rate_new:.2f}s** per lap.")