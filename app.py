"""
Customer Churn Prediction — Flask Web App
==========================================
Run this file to launch a localhost web dashboard at http://127.0.0.1:5000
"""

import os, sys, base64, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

# ── Load model artefacts once at startup ──────────────────────────────────────
model         = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))


def encode_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    df["balance_to_salary"]   = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["zero_balance_flag"]   = (df["Balance"] == 0).astype(int)
    bins   = [0, 30, 40, 50, 60, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60+"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df = pd.get_dummies(df, columns=["Geography", "age_group"])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    scale_cols = ["CreditScore","Age","Tenure","Balance","EstimatedSalary",
                  "balance_to_salary","products_per_tenure"]
    scale_cols = [c for c in scale_cols if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChurnScope — Customer Churn Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --card: #1a1a24;
    --border: #2a2a3a;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --green: #43e97b;
    --yellow: #f7971e;
    --red: #ff6584;
    --text: #e8e8f0;
    --muted: #7070a0;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Noise texture overlay ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0;
  }

  /* ── Header ── */
  header {
    padding: 2rem 3rem;
    display: flex; align-items: center; gap: 1.2rem;
    border-bottom: 1px solid var(--border);
    position: relative; z-index: 1;
    background: linear-gradient(180deg, rgba(108,99,255,0.08) 0%, transparent 100%);
  }
  .logo-dot {
    width: 12px; height: 12px;
    background: var(--accent); border-radius: 50%;
    box-shadow: 0 0 20px var(--accent);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.3)} }
  header h1 {
    font-family: var(--font-head);
    font-size: 1.5rem; font-weight: 800;
    letter-spacing: -0.02em;
  }
  header span { color: var(--accent); }
  .header-sub {
    margin-left: auto;
    font-size: 0.78rem; color: var(--muted);
    background: var(--card);
    padding: 0.4rem 1rem; border-radius: 20px;
    border: 1px solid var(--border);
  }

  /* ── Tabs ── */
  .tabs {
    display: flex; gap: 0;
    padding: 1.5rem 3rem 0;
    position: relative; z-index: 1;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 0.7rem 1.8rem;
    font-family: var(--font-head);
    font-size: 0.85rem; font-weight: 600;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ── Content ── */
  .page { display: none; padding: 2.5rem 3rem; position: relative; z-index: 1; }
  .page.active { display: block; }

  /* ── Predict Page ── */
  .predict-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 2rem; align-items: start;
  }

  .form-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
  }
  .form-card h2 {
    font-family: var(--font-head);
    font-size: 1.1rem; font-weight: 700;
    margin-bottom: 1.5rem;
    color: var(--text);
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .field { display: flex; flex-direction: column; gap: 0.4rem; }
  .field label {
    font-size: 0.75rem; font-weight: 500;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .field input, .field select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    font-family: var(--font-body);
    font-size: 0.9rem;
    transition: border-color 0.2s;
    width: 100%;
  }
  .field input:focus, .field select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15);
  }
  .field select option { background: var(--surface); }

  .btn-predict {
    width: 100%; margin-top: 1.5rem;
    padding: 0.9rem;
    background: var(--accent);
    color: white;
    border: none; border-radius: 10px;
    font-family: var(--font-head);
    font-size: 0.95rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
    position: relative; overflow: hidden;
  }
  .btn-predict:hover {
    background: #7c75ff;
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.4);
  }
  .btn-predict:active { transform: translateY(0); }

  /* ── Result Card ── */
  .result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    position: sticky; top: 2rem;
    transition: border-color 0.4s;
  }
  .result-card h2 {
    font-family: var(--font-head);
    font-size: 1rem; font-weight: 700;
    margin-bottom: 1.5rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.06em;
  }

  .result-empty {
    text-align: center; padding: 3rem 1rem;
    color: var(--muted);
  }
  .result-empty .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
  .result-empty p { font-size: 0.85rem; line-height: 1.6; }

  /* gauge */
  .gauge-wrap { text-align: center; margin-bottom: 1.5rem; }
  .gauge-label {
    font-family: var(--font-head);
    font-size: 3rem; font-weight: 800;
    line-height: 1;
  }
  .gauge-sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }

  /* ring progress */
  .ring-wrap { position: relative; width: 140px; height: 140px; margin: 0 auto 1rem; }
  .ring-wrap svg { width: 100%; height: 100%; transform: rotate(-90deg); }
  .ring-bg { fill: none; stroke: var(--border); stroke-width: 8; }
  .ring-fill { fill: none; stroke-width: 8; stroke-linecap: round;
    transition: stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1), stroke 0.4s; }
  .ring-center {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
  }
  .ring-pct { font-family: var(--font-head); font-size: 1.8rem; font-weight: 800; }
  .ring-lbl { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; }

  /* risk badge */
  .risk-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 1.2rem; border-radius: 30px;
    font-family: var(--font-head); font-size: 0.9rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 0 auto 1.5rem; display: flex; justify-content: center;
    width: fit-content; margin: 0 auto 1.5rem;
  }
  .risk-LOW    { background: rgba(67,233,123,0.15); color: var(--green); border: 1px solid rgba(67,233,123,0.3); }
  .risk-MEDIUM { background: rgba(247,151,30,0.15);  color: var(--yellow); border: 1px solid rgba(247,151,30,0.3); }
  .risk-HIGH   { background: rgba(255,101,132,0.15); color: var(--red);    border: 1px solid rgba(255,101,132,0.3); }

  .result-verdict {
    text-align: center;
    font-size: 1rem; font-weight: 500;
    padding: 1rem;
    border-radius: 10px;
    background: var(--surface);
    margin-top: 1rem;
    line-height: 1.5;
  }

  /* stats row */
  .stat-row {
    display: flex; gap: 0.8rem; margin-top: 1.2rem;
  }
  .stat-item {
    flex: 1; background: var(--surface);
    border-radius: 10px; padding: 0.8rem;
    text-align: center;
  }
  .stat-val { font-family: var(--font-head); font-size: 1.1rem; font-weight: 700; }
  .stat-lbl { font-size: 0.65rem; color: var(--muted); margin-top: 0.2rem; text-transform: uppercase; }

  /* ── Charts Page ── */
  .charts-header {
    margin-bottom: 2rem;
  }
  .charts-header h2 {
    font-family: var(--font-head);
    font-size: 1.6rem; font-weight: 800; margin-bottom: 0.4rem;
  }
  .charts-header p { color: var(--muted); font-size: 0.9rem; }

  .charts-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.5rem;
  }
  .chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }
  .chart-card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
  }
  .chart-card.wide { grid-column: span 2; }
  .chart-label {
    padding: 1rem 1.2rem 0.6rem;
    font-family: var(--font-head);
    font-size: 0.8rem; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .chart-card img {
    width: 100%; display: block;
    border-top: 1px solid var(--border);
  }

  /* ── About Page ── */
  .about-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
  .about-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.8rem;
  }
  .about-card h3 {
    font-family: var(--font-head); font-size: 1rem;
    font-weight: 700; margin-bottom: 1rem;
    color: var(--accent);
  }
  .about-card p { font-size: 0.88rem; color: var(--muted); line-height: 1.7; }
  .about-card ul { list-style: none; }
  .about-card ul li {
    font-size: 0.88rem; color: var(--muted);
    padding: 0.4rem 0; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.6rem;
  }
  .about-card ul li:last-child { border: none; }
  .dot { width:6px; height:6px; border-radius:50%; background: var(--accent); flex-shrink:0; }
  .model-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(108,99,255,0.15); color: var(--accent);
    border: 1px solid rgba(108,99,255,0.3);
    margin-top: 0.8rem; margin-right: 0.4rem;
  }

  /* loading spinner */
  .spinner {
    display: none; width: 20px; height: 20px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* responsive */
  @media (max-width: 900px) {
    .predict-grid { grid-template-columns: 1fr; }
    .charts-grid { grid-template-columns: 1fr; }
    .charts-grid .chart-card.wide { grid-column: span 1; }
    .about-grid { grid-template-columns: 1fr; }
    header { padding: 1.2rem 1.5rem; }
    .page { padding: 1.5rem; }
    .tabs { padding: 1rem 1.5rem 0; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>Churn<span>Scope</span></h1>
  <div class="header-sub">CodSoft ML Internship · Task 3</div>
</header>

<div class="tabs">
  <div class="tab active" onclick="showTab('predict')">🎯 Predict</div>
  <div class="tab" onclick="showTab('charts')">📊 Analysis Charts</div>
  <div class="tab" onclick="showTab('about')">ℹ️ About</div>
</div>

<!-- ═══════════════════ PREDICT PAGE ═══════════════════ -->
<div class="page active" id="tab-predict">
  <div class="predict-grid">

    <div class="form-card">
      <h2>Enter Customer Details</h2>
      <div class="form-grid">

        <div class="field">
          <label>Credit Score</label>
          <input type="number" id="CreditScore" value="650" min="300" max="850">
        </div>
        <div class="field">
          <label>Age</label>
          <input type="number" id="Age" value="35" min="18" max="92">
        </div>
        <div class="field">
          <label>Geography</label>
          <select id="Geography">
            <option>France</option>
            <option>Germany</option>
            <option>Spain</option>
          </select>
        </div>
        <div class="field">
          <label>Gender</label>
          <select id="Gender">
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>
        <div class="field">
          <label>Tenure (years)</label>
          <input type="number" id="Tenure" value="5" min="0" max="10">
        </div>
        <div class="field">
          <label>Account Balance (₹/€)</label>
          <input type="number" id="Balance" value="75000" min="0">
        </div>
        <div class="field">
          <label>Number of Products</label>
          <select id="NumOfProducts">
            <option value="1">1</option>
            <option value="2" selected>2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>
        <div class="field">
          <label>Estimated Salary</label>
          <input type="number" id="EstimatedSalary" value="95000" min="0">
        </div>
        <div class="field">
          <label>Has Credit Card?</label>
          <select id="HasCrCard">
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </div>
        <div class="field">
          <label>Active Member?</label>
          <select id="IsActiveMember">
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </div>

      </div>
      <button class="btn-predict" onclick="predict()">
        <span id="btn-text">Run Prediction</span>
        <div class="spinner" id="spinner"></div>
      </button>
    </div>

    <!-- Result -->
    <div class="result-card" id="result-card">
      <h2>Result</h2>
      <div class="result-empty" id="result-empty">
        <div class="icon">🔮</div>
        <p>Fill in the customer details on the left and click <strong>Run Prediction</strong> to see the churn risk analysis.</p>
      </div>
      <div id="result-content" style="display:none">
        <div class="ring-wrap">
          <svg viewBox="0 0 100 100">
            <circle class="ring-bg" cx="50" cy="50" r="42"/>
            <circle class="ring-fill" id="ring" cx="50" cy="50" r="42"
              stroke-dasharray="263.9" stroke-dashoffset="263.9"/>
          </svg>
          <div class="ring-center">
            <div class="ring-pct" id="ring-pct">0%</div>
            <div class="ring-lbl">churn risk</div>
          </div>
        </div>

        <div style="text-align:center; margin-bottom:1rem">
          <div class="risk-badge" id="risk-badge">—</div>
        </div>

        <div class="result-verdict" id="verdict"></div>

        <div class="stat-row">
          <div class="stat-item">
            <div class="stat-val" id="stat-prob">—</div>
            <div class="stat-lbl">Probability</div>
          </div>
          <div class="stat-item">
            <div class="stat-val" id="stat-decision">—</div>
            <div class="stat-lbl">Decision</div>
          </div>
          <div class="stat-item">
            <div class="stat-val" id="stat-risk">—</div>
            <div class="stat-lbl">Risk Level</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════ CHARTS PAGE ═══════════════════ -->
<div class="page" id="tab-charts">
  <div class="charts-header">
    <h2>Exploratory Analysis & Model Results</h2>
    <p>All charts are auto-generated from the training data — no manual editing.</p>
  </div>
  <div class="charts-grid">
    {% for fname, label, wide in charts %}
    <div class="chart-card {{ 'wide' if wide else '' }}">
      <div class="chart-label">{{ label }}</div>
      {% if fname %}
      <img src="/chart/{{ fname }}" alt="{{ label }}">
      {% else %}
      <div style="padding:2rem;color:var(--muted);text-align:center;font-size:0.85rem">Chart not generated yet — run python main.py first</div>
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>

<!-- ═══════════════════ ABOUT PAGE ═══════════════════ -->
<div class="page" id="tab-about">
  <div class="about-grid">
    <div class="about-card">
      <h3>What This Project Does</h3>
      <p>This machine learning model predicts whether a bank customer is likely to close their account (churn) based on their profile. Banks use this to proactively reach out with retention offers before a customer leaves.</p>
      <br>
      <p>The model was trained on 10,000 customer records with features like age, balance, geography, activity status, and number of products held.</p>
    </div>
    <div class="about-card">
      <h3>Models Trained</h3>
      <ul>
        <li><span class="dot"></span> <strong>Logistic Regression</strong> — Interpretable linear baseline</li>
        <li><span class="dot"></span> <strong>Random Forest</strong> — 200 decision trees, majority vote</li>
        <li><span class="dot"></span> <strong>Gradient Boosting</strong> — Sequential error correction</li>
      </ul>
      <div style="margin-top:1rem">
        <span class="model-badge">Best: Logistic Regression</span>
        <span class="model-badge">ROC-AUC: 0.64</span>
      </div>
    </div>
    <div class="about-card">
      <h3>Key Churn Risk Factors</h3>
      <ul>
        <li><span class="dot"></span> Age over 50 — highest predictor of churn</li>
        <li><span class="dot"></span> Inactive membership status</li>
        <li><span class="dot"></span> Geography = Germany (2× higher churn rate)</li>
        <li><span class="dot"></span> Holding 3 or 4 products (counter-intuitive)</li>
        <li><span class="dot"></span> Zero account balance</li>
      </ul>
    </div>
    <div class="about-card">
      <h3>Why ROC-AUC and Not Accuracy?</h3>
      <p>Only ~17% of customers churn. A model that always predicts "won't churn" gets 83% accuracy but is completely useless.</p>
      <br>
      <p>ROC-AUC measures how well the model <em>ranks</em> churners above non-churners — a score of 0.64 means the model correctly ranks a churner above a non-churner 64% of the time, versus 50% for random guessing.</p>
    </div>
    <div class="about-card" style="grid-column: span 2">
      <h3>Risk Level Guide</h3>
      <ul>
        <li><span class="dot" style="background:var(--green)"></span> <strong style="color:var(--green)">Low Risk (&lt;30%)</strong> — Customer is likely to stay. No action needed.</li>
        <li><span class="dot" style="background:var(--yellow)"></span> <strong style="color:var(--yellow)">Medium Risk (30–60%)</strong> — Worth sending a retention offer or checking in.</li>
        <li><span class="dot" style="background:var(--red)"></span> <strong style="color:var(--red)">High Risk (&gt;60%)</strong> — Priority intervention. Assign a relationship manager.</li>
      </ul>
    </div>
  </div>
</div>

<script>
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

async function predict() {
  const btnText = document.getElementById('btn-text');
  const spinner = document.getElementById('spinner');
  btnText.style.display = 'none';
  spinner.style.display = 'block';

  const payload = {
    CreditScore:     +document.getElementById('CreditScore').value,
    Geography:        document.getElementById('Geography').value,
    Gender:           document.getElementById('Gender').value,
    Age:             +document.getElementById('Age').value,
    Tenure:          +document.getElementById('Tenure').value,
    Balance:         +document.getElementById('Balance').value,
    NumOfProducts:   +document.getElementById('NumOfProducts').value,
    HasCrCard:       +document.getElementById('HasCrCard').value,
    IsActiveMember:  +document.getElementById('IsActiveMember').value,
    EstimatedSalary: +document.getElementById('EstimatedSalary').value,
  };

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    showResult(data);
  } catch(e) {
    alert('Prediction failed: ' + e);
  } finally {
    btnText.style.display = 'inline';
    spinner.style.display = 'none';
  }
}

function showResult(data) {
  document.getElementById('result-empty').style.display   = 'none';
  document.getElementById('result-content').style.display = 'block';

  const pct   = Math.round(data.churn_probability * 100);
  const circ  = 263.9;
  const offset = circ - (circ * data.churn_probability);

  const colorMap = { LOW: '#43e97b', MEDIUM: '#f7971e', HIGH: '#ff6584' };
  const color = colorMap[data.risk_level] || '#6c63ff';

  // Ring animation
  const ring = document.getElementById('ring');
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = color;

  document.getElementById('ring-pct').textContent = pct + '%';
  document.getElementById('ring-pct').style.color = color;

  // Risk badge
  const badge = document.getElementById('risk-badge');
  badge.textContent = data.risk_level + ' RISK';
  badge.className = 'risk-badge risk-' + data.risk_level;

  // Verdict
  const verdicts = {
    LOW:    '✅ This customer is <strong>likely to stay</strong>. No immediate action required.',
    MEDIUM: '⚠️ This customer shows <strong>moderate churn signals</strong>. Consider a retention offer.',
    HIGH:   '🚨 This customer is at <strong>high risk of churning</strong>. Prioritise intervention now.'
  };
  document.getElementById('verdict').innerHTML = verdicts[data.risk_level];

  // Stats
  document.getElementById('stat-prob').textContent     = (data.churn_probability * 100).toFixed(1) + '%';
  document.getElementById('stat-decision').textContent = data.will_churn ? 'Churn' : 'Stay';
  document.getElementById('stat-decision').style.color = data.will_churn ? '#ff6584' : '#43e97b';
  document.getElementById('stat-risk').textContent     = data.risk_level;
  document.getElementById('stat-risk').style.color     = color;
}
</script>

</body>
</html>
"""

@app.route("/")
def index():
    chart_list = [
        ("01_churn_distribution.png",    "Churn Distribution",              False),
        ("02_numeric_distributions.png", "Feature Distributions by Churn",  True),
        ("03_categorical_churn_rates.png","Churn Rate by Category",          True),
        ("04_correlation_heatmap.png",   "Correlation Heatmap",             False),
        ("05_age_balance_scatter.png",   "Age vs Balance Scatter",          False),
        ("06_confusion_matrices.png",    "Confusion Matrices",              True),
        ("07_roc_curves.png",            "ROC Curves — All Models",         False),
        ("08_precision_recall.png",      "Precision-Recall Curves",         False),
        ("09_feature_importance.png",    "Feature Importance",              False),
        ("10_model_comparison.png",      "Model Comparison",                False),
    ]
    # Only include charts that actually exist
    charts = []
    for fname, label, wide in chart_list:
        exists = os.path.exists(os.path.join(PLOTS_DIR, fname))
        charts.append((fname if exists else None, label, wide))

    return render_template_string(HTML_TEMPLATE, charts=charts)


@app.route("/chart/<filename>")
def serve_chart(filename):
    from flask import send_from_directory
    return send_from_directory(PLOTS_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    try:
        X = preprocess_input(data)
        prob       = float(model.predict_proba(X)[0][1])
        will_churn = prob >= 0.5
        if prob < 0.30:   risk = "LOW"
        elif prob < 0.60: risk = "MEDIUM"
        else:             risk = "HIGH"

        return jsonify({
            "churn_probability": round(prob, 4),
            "will_churn":        will_churn,
            "risk_level":        risk
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ChurnScope — Web Dashboard")
    print("="*55)
    print("  Open your browser and go to:")
    print("  → http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
