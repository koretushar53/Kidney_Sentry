"""
KidneyScan-AI — Frontend (index.py)
====================================
Serves the dashboard as a single self-contained HTML page (styled
close to the original platform's look) with two clearly separated
CT sections, per spec:

  SECTION 1: CLASSIFICATION (always shown after a CT upload)
  SECTION 2: TUMOR MEASUREMENT (only shown when Model 1 says Tumor)

app.py imports DASHBOARD_HTML and serves it at '/'.
"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KidneyScan-AI</title>
<style>
    :root {
        --primary: #1e3a8a;
        --primary-light: #2563eb;
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --text: #1e293b;
        --border: #e2e8f0;
        --danger: #ef4444;
        --warn: #f97316;
        --ok: #22c55e;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: var(--bg);
        color: var(--text);
        margin: 0;
        padding: 30px 20px;
    }
    .container { max-width: 900px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 30px; }
    h1 { color: var(--primary); margin: 0 0 5px 0; }
    .subtitle { color: #64748b; font-size: 0.95rem; }
    .card {
        background: var(--card-bg); padding: 24px; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;
        border: 1px solid var(--border);
    }
    h2 {
        margin-top: 0; color: var(--primary); font-size: 1.25rem;
        border-bottom: 1px solid var(--border); padding-bottom: 10px;
    }
    label { display: block; font-size: 0.85rem; margin: 12px 0 4px; color: #475569; }
    input[type=number], input[type=file] {
        width: 100%; padding: 8px 10px; border: 1px solid var(--border);
        border-radius: 6px; box-sizing: border-box; font-size: 0.95rem;
    }
    .row { display: flex; gap: 16px; flex-wrap: wrap; }
    .row > div { flex: 1; min-width: 140px; }
    .checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 14px; }
    button {
        background: var(--primary-light); color: white; border: none;
        padding: 10px 18px; border-radius: 6px; font-size: 0.95rem;
        cursor: pointer; margin-top: 16px;
    }
    button:hover { background: var(--primary); }
    .hidden { display: none; }
    .result-box {
        margin-top: 16px; padding: 14px; border-radius: 8px;
        background: #f1f5f9; font-size: 0.9rem;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; color: white;
    }
    .badge-tumor { background: var(--danger); }
    .badge-stone { background: var(--warn); }
    .badge-cyst { background: #eab308; }
    .badge-normal { background: var(--ok); }
    .img-row { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
    .img-row img { max-width: 260px; border-radius: 6px; border: 1px solid var(--border); }
    .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px; }
    .metric { background: white; border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
    .metric .label { font-size: 0.75rem; color: #64748b; }
    .metric .value { font-size: 1.1rem; font-weight: 600; color: var(--primary); }
    .error { color: var(--danger); font-size: 0.9rem; margin-top: 8px; }
    .note { color: #64748b; font-size: 0.8rem; margin-top: 8px; font-style: italic; }
    .guidance-block { margin-bottom: 14px; }
    .guidance-block strong { display: block; color: var(--primary); margin-bottom: 5px; }
    .guidance-block ul, .guidance-block ol { margin: 5px 0 0 20px; padding: 0; }
    .guidance-block li { margin: 4px 0; }
    .guidance-important { border-top: 1px solid var(--border); padding-top: 12px; font-weight: 600; }
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>KidneyScan-AI</h1>
        <div class="subtitle">Lab risk triage &middot; CT classification &middot; tumor measurement</div>
    </header>

    <!-- LAB RISK -->
    <div class="card">
        <h2>1. Urine / Blood Report</h2>
        <div class="row">
            <div>
                <label>Creatinine (mg/dL)</label>
                <input type="number" step="0.01" id="creatinine" value="1.0">
            </div>
            <div>
                <label>eGFR (mL/min/1.73m&sup2;)</label>
                <input type="number" step="1" id="egfr" value="90">
            </div>
            <div>
                <label>Urine Protein (mg/dL)</label>
                <input type="number" step="1" id="urineProtein" value="0">
            </div>
        </div>
        <div class="checkbox-row">
            <input type="checkbox" id="hematuria">
            <label style="margin:0;">Hematuria present</label>
        </div>
        <button onclick="assessRisk()">Assess Risk</button>
        <div id="riskResult" class="result-box hidden"></div>
    </div>

    <!-- CT UPLOAD -->
    <div class="card">
        <h2>2. CT Scan</h2>
        <label>Upload CT (.png, .jpg, .jpeg, .nii, .nii.gz)</label>
        <input type="file" id="ctFile" accept=".png,.jpg,.jpeg,.nii,.nii.gz">
        <button onclick="analyzeCT()">Analyze CT</button>
        <div id="ctError" class="error hidden"></div>
    </div>

    <!-- SECTION 1: CLASSIFICATION -->
    <div class="card hidden" id="section1">
        <h2>Section 1 &mdash; CT Classification</h2>
        <div id="classBadge"></div>
        <div class="metric-grid">
            <div class="metric"><div class="label">Prediction</div><div class="value" id="classOutput">-</div></div>
            <div class="metric"><div class="label">Confidence</div><div class="value" id="confOutput">-</div></div>
        </div>
        <div class="img-row">
            <div><div class="note">Annotated (Red Rectangle)</div><img id="annotatedImage"></div>
        </div>
        <div class="note" id="locOutput"></div>
    </div>

    <!-- SECTION 2: TUMOR MEASUREMENT -->
    <div class="card hidden" id="section2">
        <h2>Section 2 &mdash; Tumor Analysis</h2>
        <div id="segStatus" class="note"></div>
        <div class="metric-grid" id="measurementGrid"></div>
        <div class="note" id="unitsNote"></div>
    </div>

    <!-- RULE-BASED GUIDANCE -->
    <div class="card hidden" id="guidanceCard">
        <h2>Guidance</h2>
        <div id="guidanceText" class="result-box">
            <div class="guidance-block"><strong>Tumor Size Assessment</strong><div id="guidanceSizeCategory"></div><div id="guidanceSizeMessage"></div></div>
            <div class="guidance-block"><strong>Location</strong><div id="guidanceLocation"></div></div>
            <div class="guidance-block"><strong>AI Confidence</strong><div id="guidanceConfidence"></div></div>
            <div class="guidance-block"><strong>Segmentation</strong><div id="guidanceSegmentation"></div></div>
            <div class="guidance-block"><strong>Precautions</strong><ul id="guidancePrecautions"></ul></div>
            <div class="guidance-block"><strong>Recommended Next Steps</strong><ol id="guidanceNextSteps"></ol></div>
            <div class="guidance-important" id="guidanceImportant"></div>
        </div>
    </div>

    <!-- RULE-BASED DIET RECOMMENDATION -->
    <div class="card hidden" id="dietCard">
        <h2>Diet Recommendation</h2>
        <div class="result-box">
            <div class="guidance-block"><strong id="dietTitle"></strong></div>
            <div class="guidance-block"><strong>Foods to Include</strong><ul id="dietInclude"></ul></div>
            <div class="guidance-block"><strong>Foods to Limit</strong><ul id="dietLimit"></ul></div>
            <div class="guidance-block"><strong>General Recommendations</strong><ul id="dietRecommendations"></ul></div>
            <div class="guidance-important"><strong>Important Note</strong><div id="dietNote"></div></div>
        </div>
    </div>
</div>

<script>
async function assessRisk() {
    const payload = {
        creatinine: parseFloat(document.getElementById('creatinine').value),
        egfr: parseFloat(document.getElementById('egfr').value),
        urine_protein: parseFloat(document.getElementById('urineProtein').value),
        hematuria: document.getElementById('hematuria').checked
    };
    const res = await fetch('/api/assess-risk', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    const box = document.getElementById('riskResult');
    box.classList.remove('hidden');
    box.innerHTML = `<strong>${data.risk_category}</strong><br>${data.message}`;
}

function badgeClass(label) {
    return {
        'Tumor': 'badge-tumor', 'Stone': 'badge-stone',
        'Cyst': 'badge-cyst', 'Normal': 'badge-normal'
    }[label] || 'badge-normal';
}

function metricCard(label, value) {
    return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

async function analyzeCT() {
    const fileInput = document.getElementById('ctFile');
    const errorBox = document.getElementById('ctError');
    errorBox.classList.add('hidden');
    document.getElementById('section1').classList.add('hidden');
    document.getElementById('section2').classList.add('hidden');
    document.getElementById('guidanceCard').classList.add('hidden');
    document.getElementById('dietCard').classList.add('hidden');

    if (!fileInput.files.length) {
        errorBox.textContent = 'Please choose a CT file first.';
        errorBox.classList.remove('hidden');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const res = await fetch('/api/analyze-ct', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
        errorBox.textContent = data.error;
        errorBox.classList.remove('hidden');
        return;
    }

    // --- Section 1 ---
    const c = data.classification;
    document.getElementById('section1').classList.remove('hidden');
    document.getElementById('classBadge').innerHTML =
        `<span class="badge ${badgeClass(c.label)}">${c.label}</span>`;
    document.getElementById('classOutput').textContent = c.label;
    document.getElementById('confOutput').textContent = (c.confidence * 100).toFixed(1) + '%';
    document.getElementById('annotatedImage').src = c.annotated_image;
    document.getElementById('locOutput').textContent =
        'Frame-relative region: ' + c.frame_location;

    // --- Section 2 (Tumor only) ---
    if (c.label === 'Tumor') {
        const seg = data.segmentation;
        const m = data.measurements || {};
        const section2 = document.getElementById('section2');
        section2.classList.remove('hidden');

        const status = document.getElementById('segStatus');
        if (!seg.performed) {
            status.textContent = 'Segmentation could not be completed.';
        } else if (!seg.tumor_found) {
            status.textContent = 'Segmentation completed — no tumor region detected by Model 2.';
        } else {
            status.textContent = `Segmentation completed — ${seg.components} component(s) found.`;
        }

        const grid = document.getElementById('measurementGrid');
        grid.innerHTML = '';
        if (m.volume_ml !== undefined) grid.innerHTML += metricCard('Volume', m.volume_ml + ' mL');
        if (m.maximum_diameter_mm !== undefined) grid.innerHTML += metricCard('Max Diameter', m.maximum_diameter_mm + ' mm');
        if (m.dimensions_mm) grid.innerHTML += metricCard('Dimensions', m.dimensions_mm.join(' x ') + ' mm');
        if (m.location) grid.innerHTML += metricCard('Location', m.location);
        if (m.area_pixels !== undefined) grid.innerHTML += metricCard('Area', m.area_pixels + ' px');
        if (m.maximum_diameter_pixels !== undefined) grid.innerHTML += metricCard('Max Diameter', m.maximum_diameter_pixels + ' px');

        document.getElementById('unitsNote').textContent = m.units_note || '';
    }

    // --- Deterministic rule-based guidance ---
    if (c.label === 'Tumor' && data.guidance && data.guidance.tumor_size_assessment) {
        const guidance = data.guidance;
        const size = guidance.tumor_size_assessment;
        document.getElementById('guidanceCard').classList.remove('hidden');
        document.getElementById('guidanceSizeCategory').textContent = size.category;
        document.getElementById('guidanceSizeMessage').textContent = size.message;
        document.getElementById('guidanceLocation').textContent = guidance.location;
        document.getElementById('guidanceConfidence').textContent = guidance.confidence;
        document.getElementById('guidanceSegmentation').textContent = guidance.segmentation;
        document.getElementById('guidancePrecautions').innerHTML = guidance.precautions
            .map(item => `<li>${item}</li>`).join('');
        document.getElementById('guidanceNextSteps').innerHTML = guidance.next_steps
            .map(item => `<li>${item}</li>`).join('');
        document.getElementById('guidanceImportant').textContent = guidance.important;
    }

    // --- Deterministic rule-based diet recommendation ---
    if (data.diet_recommendation) {
        const diet = data.diet_recommendation;
        const listItems = (items) => items.map(item => `<li>${item}</li>`).join('');
        document.getElementById('dietCard').classList.remove('hidden');
        document.getElementById('dietTitle').textContent = diet.title;
        document.getElementById('dietInclude').innerHTML = listItems(diet.foods_to_include);
        document.getElementById('dietLimit').innerHTML = listItems(diet.foods_to_limit);
        document.getElementById('dietRecommendations').innerHTML = listItems(diet.recommendations);
        document.getElementById('dietNote').textContent = diet.note;
    }
}
</script>
</body>
</html>
"""