# generate_dashboard.py
# ── Reads from metrics.db and outputs a fully connected HTML dashboard ─────
# Run: python generate_dashboard.py
# Output: outputs/latest/dashboard.html + outputs/archive/YYYY-MM-DD/dashboard.html

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
import pipeline_config as cfg


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metrics.db")
OUTPUT_LATEST  = os.path.join(cfg.OUTPUTS_LATEST, "dashboard.html")
OUTPUT_ARCHIVE = os.path.join(cfg.OUTPUTS_ARCHIVE, datetime.now().strftime("%Y-%m-%d"), "dashboard.html")

def load_data():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        print("Run the pipeline first, or run: python aggregation_db.py rebuild")
        return None
    conn = sqlite3.connect(DB_PATH)
    def safe_read(sql):
        try: return pd.read_sql(sql, conn)
        except: return pd.DataFrame()
    summary  = safe_read("SELECT * FROM monthly_summary ORDER BY run_date")
    tags     = safe_read("SELECT * FROM tag_group_monthly ORDER BY run_date, tag_group")
    features = safe_read("SELECT * FROM feature_monthly ORDER BY run_date")
    errors   = safe_read("SELECT * FROM error_monthly ORDER BY run_date, count DESC")
    conn.close()
    return summary, tags, features, errors

def safe_val(val, default=0):
    if val is None: return default
    try:
        import math
        if math.isnan(float(val)): return default
    except: pass
    return val

def build_months_js(summary, tags, features, errors):
    months = []
    for _, row in summary.iterrows():
        run_date    = row["run_date"]
        month_label = pd.to_datetime(run_date).strftime("%b %Y")
        # ratings stored as absolute counts in DB (rating_1 .. rating_5)
        ratings = {s: int(safe_val(row.get(f"rating_{s}"), 0)) for s in [1,2,3,4,5]}
        total_rated = sum(ratings.values())
        pos_pct = float(safe_val(row.get("positive_pct"), 0))
        neg_pct = float(safe_val(row.get("negative_pct"), 0))
        neu_pct = round(max(0, 100 - pos_pct - neg_pct), 1)
        nsat    = round(float(safe_val(row.get("nsat"), 0)), 1)

        month_tags = tags[tags["run_date"] == run_date] if not tags.empty else pd.DataFrame()
        tag_list = []
        if not month_tags.empty:
            for _, tr in month_tags.sort_values("total_reviews", ascending=False).iterrows():
                tag_list.append({
                    "t": str(tr["tag_group"]).strip(),
                    "n": int(safe_val(tr.get("total_reviews"), 0)),
                    "s": round(float(safe_val(tr.get("avg_sentiment_score"), 0)), 3)
                })

        month_feat = features[features["run_date"] == run_date] if not features.empty else pd.DataFrame()
        feat_list = []
        if not month_feat.empty:
            excl = {"Planning Portal", "", "nan"}
            for _, fr in month_feat[~month_feat["feature"].isin(excl)].sort_values("negative_pct", ascending=False).head(10).iterrows():
                feat_list.append({
                    "f": str(fr["feature"]),
                    "n": int(safe_val(fr.get("total_mentions"), 0)),
                    "neg_pct": float(safe_val(fr.get("negative_pct"), 0))
                })

        month_err = errors[errors["run_date"] == run_date] if not errors.empty else pd.DataFrame()
        err_list = []
        if not month_err.empty:
            for _, er in month_err.sort_values("count", ascending=False).head(8).iterrows():
                err_list.append({"e": str(er["error_pattern"]), "n": int(safe_val(er.get("count"), 0))})

        months.append({
            "label":      month_label,
            "run_date":   run_date,
            "total":      int(safe_val(row.get("total_respondents"), 0)),

            "feedback":   int(safe_val(row.get("total_with_feedback"), 0)),
            "avgRating":  round(float(safe_val(row.get("avg_rating"), 0)), 1),
            "nsat":       nsat,
            "ratings":    ratings,      # absolute counts e.g. {5: 412, 4: 180, ...}
            "totalRated": total_rated,  # sum of rating counts for scaling bars
            "pos":        pos_pct,
            "neu":        neu_pct,
            "neg":        neg_pct,
            "tags":       tag_list,
            "features":   feat_list,
            "errors":     err_list,
        })
    return months

def build_dashboard(months):
    months_json = json.dumps(months, ensure_ascii=False)
    generated   = datetime.now().strftime("%d %B %Y, %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planning Portal — Feedback Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#f5f6f8;--card:#fff;--border:#e4e6ea;--text:#1a1a1a;--muted:#6b7280;--hint:#9ca3af;--blue:#185FA5;--light-blue:#E6F1FB;--green:#1D9E75;--red:#E24B4A;--amber:#BA7517}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}}
.wrap{{max-width:1280px;margin:0 auto;padding:24px 28px}}
.page-header{{background:var(--card);border-bottom:1px solid var(--border);padding:18px 28px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100}}
.page-header h1{{font-size:17px;font-weight:600}}
.meta{{font-size:12px;color:var(--muted)}}
.tab-nav{{display:flex;gap:0;border-bottom:2px solid var(--border);margin-bottom:24px;margin-top:8px}}
.tab-btn{{padding:10px 22px;font-size:13px;font-weight:500;color:var(--muted);background:none;border:none;border-bottom:2px solid transparent;margin-bottom:-2px;cursor:pointer;transition:all .15s}}
.tab-btn:hover{{color:var(--blue)}}
.tab-btn.active{{color:var(--blue);border-bottom-color:var(--blue);font-weight:600}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}
.section-title{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--hint);margin:28px 0 12px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:6px}}
.metric{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}}
.metric-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}}
.metric-value{{font-size:28px;font-weight:600;line-height:1}}
.metric-sub{{font-size:11px;color:var(--hint);margin-top:4px}}
.metric-value.pos{{color:var(--green)}}.metric-value.neg{{color:var(--red)}}.metric-value.amber{{color:var(--amber)}}.metric-value.blue{{color:var(--blue)}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px}}
.card-title{{font-size:13px;font-weight:600;margin-bottom:3px}}
.card-sub{{font-size:11px;color:var(--muted);margin-bottom:14px}}
.rag-labels{{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:5px}}
.rag-bar{{display:flex;border-radius:6px;overflow:hidden;height:28px}}
.rag-seg{{display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;transition:flex .3s}}
.star-row{{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px}}
.star-label{{width:22px;color:var(--muted);flex-shrink:0;font-size:11px}}
.star-track{{flex:1;height:11px;background:#f1f2f4;border-radius:3px;overflow:hidden}}
.star-fill{{height:100%;border-radius:3px;transition:width .3s}}
.star-count{{width:42px;text-align:right;color:var(--muted);font-size:11px;flex-shrink:0}}
.tag-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.tag-label{{width:190px;flex-shrink:0;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11px}}
.tag-track{{flex:1;height:12px;background:#f1f2f4;border-radius:3px;overflow:hidden}}
.tag-bar{{height:100%;border-radius:3px;transition:width .3s}}
.tag-count{{width:36px;text-align:right;color:var(--hint);font-size:11px;flex-shrink:0}}
.month-pills{{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0}}
.pill{{padding:4px 12px;border-radius:20px;background:var(--card);border:1px solid var(--border);font-size:11px;color:var(--muted);cursor:pointer;transition:all .15s}}
.pill:hover{{border-color:var(--blue);color:var(--blue)}}
.pill.active{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.agg-filter{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:16px}}
.agg-filter-title{{font-size:13px;font-weight:600;margin-bottom:10px}}
.agg-month-grid{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
.agg-pill{{padding:4px 12px;border-radius:20px;background:var(--bg);border:1px solid var(--border);font-size:11px;color:var(--muted);cursor:pointer;transition:all .15s;user-select:none}}
.agg-pill:hover{{border-color:var(--blue);color:var(--blue)}}
.agg-pill.selected{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.agg-actions{{display:flex;gap:8px}}
.agg-btn{{padding:5px 14px;border-radius:6px;font-size:12px;cursor:pointer;border:1px solid var(--border);background:var(--card);color:var(--text);transition:all .15s}}
.agg-btn:hover{{border-color:var(--blue);color:var(--blue)}}
.agg-btn.primary{{background:var(--blue);color:#fff;border-color:var(--blue)}}
.agg-btn.primary:hover{{background:#1450a0}}
.empty-state{{padding:32px;text-align:center;color:var(--hint);font-size:13px}}
@media(max-width:800px){{.metrics,.grid-2{{grid-template-columns:1fr}}.wrap{{padding:16px}}}}
</style>
</head>
<body>
<div class="page-header">
  <div><h1>Planning Portal — user feedback dashboard</h1><div class="meta" id="headerMeta">Loading...</div></div>
  <div class="meta">Generated: {generated}</div>
</div>
<div class="wrap">

  <div class="tab-nav">
    <button class="tab-btn active" onclick="switchTab('monthly',this)">Monthly view</button>
    <button class="tab-btn" onclick="switchTab('aggregate',this)">Aggregate view</button>
  </div>

  <!-- ═══ MONTHLY TAB ═══════════════════════════════════════════════════ -->
  <div class="tab-panel active" id="tab-monthly">
    <div class="month-pills" id="pills"></div>
    <div class="section-title">This month at a glance</div>
    <div class="metrics">
      <div class="metric"><div class="metric-label">Total responses</div><div class="metric-value blue" id="mTotal">—</div><div class="metric-sub" id="mTotalSub">—</div></div>
      <div class="metric"><div class="metric-label">Average rating</div><div class="metric-value amber" id="mRating">—</div><div class="metric-sub" id="mRatingSub">—</div></div>
      <div class="metric"><div class="metric-label">NSAT score</div><div class="metric-value pos" id="mNsat">—</div><div class="metric-sub" id="mNsatSub">—</div></div>
      <div class="metric"><div class="metric-label">Left feedback</div><div class="metric-value" id="mFeedback">—</div><div class="metric-sub" id="mFeedbackPct">—</div></div>
    </div>
    <div class="section-title">Ratings</div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Rating distribution — this month</div>
        <div class="card-sub" id="ratingDistSub"></div>
        <div id="starBars"></div>
      </div>
      <div class="card">
        <div class="card-title">Average rating — all months</div>
        <div class="card-sub">Trend over time</div>
        <div style="position:relative;height:190px"><canvas id="avgRatingChart"></canvas></div>
      </div>
    </div>
    <div class="section-title">Sentiment — comments only</div>
    <div class="card" style="margin-bottom:12px">
      <div class="card-title">Overall comment sentiment — this month</div>
      <div class="card-sub">Red = negative · amber = neutral · green = positive</div>
      <div class="rag-labels"><span id="ragNegLabel">—</span><span id="ragNeuLabel">—</span><span id="ragPosLabel">—</span></div>
      <div class="rag-bar" id="ragBar"></div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-title">NSAT score — all months</div><div class="card-sub">Net Satisfaction Score trend</div><div style="position:relative;height:190px"><canvas id="nsatChart"></canvas></div></div>
      <div class="card"><div class="card-title">Comment sentiment — all months</div><div class="card-sub">% positive vs negative each month</div><div style="position:relative;height:190px"><canvas id="sentChart"></canvas></div></div>
    </div>
    <div class="section-title">Thematic analysis</div>
    <div class="grid-2">
      <div class="card"><div class="card-title">Tag group mentions — this month</div><div class="card-sub">Volume of comments per theme</div><div id="tagMentions"></div></div>
      <div class="card"><div class="card-title">Tag group sentiment — this month</div><div class="card-sub">Sentiment score per theme (−1 = all negative, +1 = all positive)</div><div style="position:relative" id="tagSentWrap"><canvas id="tagSentChart"></canvas></div></div>
    </div>
    <div class="section-title">Feature & error analysis</div>
    <div class="grid-2">
      <div class="card"><div class="card-title">Most negatively mentioned features</div><div class="card-sub">% of mentions that are negative</div><div style="position:relative" id="featWrap"><canvas id="featChart"></canvas></div></div>
      <div class="card"><div class="card-title">Top error patterns</div><div class="card-sub">Most frequently reported issues</div><div style="position:relative" id="errorWrap"><canvas id="errorChart"></canvas></div></div>
    </div>
  </div>

  <!-- ═══ AGGREGATE TAB ════════════════════════════════════════════════ -->
  <div class="tab-panel" id="tab-aggregate">
    <div class="agg-filter">
      <div class="agg-filter-title">Select months to include in aggregate</div>
      <div class="agg-month-grid" id="aggPills"></div>
      <div class="agg-actions">
        <button class="agg-btn" onclick="aggSelectAll()">Select all</button>
        <button class="agg-btn" onclick="aggClearAll()">Clear all</button>
        <button class="agg-btn primary" onclick="updateAggregate()">Update</button>
      </div>
    </div>
    <div class="section-title">Aggregate at a glance</div>
    <div class="metrics">
      <div class="metric"><div class="metric-label">Total responses</div><div class="metric-value blue" id="aTotal">—</div><div class="metric-sub" id="aTotalSub">—</div></div>
      <div class="metric"><div class="metric-label">Average rating</div><div class="metric-value amber" id="aRating">—</div><div class="metric-sub">across selected months</div></div>
      <div class="metric"><div class="metric-label">Positive sentiment</div><div class="metric-value pos" id="aPos">—</div><div class="metric-sub" id="aPosSub">—</div></div>
      <div class="metric"><div class="metric-label">Negative sentiment</div><div class="metric-value neg" id="aNeg">—</div><div class="metric-sub" id="aNegSub">—</div></div>
    </div>
    <div class="section-title">Sentiment</div>
    <div class="card" style="margin-bottom:12px">
      <div class="card-title">Overall comment sentiment — aggregate</div>
      <div class="card-sub">Red = negative · amber = neutral · green = positive</div>
      <div class="rag-labels"><span id="aRagNegLabel">—</span><span id="aRagNeuLabel">—</span><span id="aRagPosLabel">—</span></div>
      <div class="rag-bar" id="aRagBar"></div>
    </div>
    <div class="section-title">Ratings</div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Rating distribution — aggregate</div>
        <div class="card-sub" id="aRatingDistSub"></div>
        <div id="aStarBars"></div>
      </div>
      <div class="card">
        <div class="card-title">Average rating by month</div>
        <div class="card-sub">Selected months highlighted</div>
        <div style="position:relative;height:190px"><canvas id="aAvgRatingChart"></canvas></div>
      </div>
    </div>
    <div class="section-title">Thematic analysis</div>
    <div class="grid-2">
      <div class="card"><div class="card-title">Tag group mentions — aggregate</div><div class="card-sub">Total volume across selected months</div><div id="aTagMentions"></div></div>
      <div class="card"><div class="card-title">Tag group sentiment — aggregate</div><div class="card-sub">Weighted average sentiment score per theme</div><div style="position:relative" id="aTagSentWrap"><canvas id="aTagSentChart"></canvas></div></div>
    </div>
    <div class="section-title">Feature & error analysis</div>
    <div class="grid-2">
      <div class="card"><div class="card-title">Most negatively mentioned features</div><div class="card-sub">% of mentions that are negative across selected months</div><div style="position:relative" id="aFeatWrap"><canvas id="aFeatChart"></canvas></div></div>
      <div class="card"><div class="card-title">Top error patterns</div><div class="card-sub">Most frequently reported issues across selected months</div><div style="position:relative" id="aErrorWrap"><canvas id="aErrorChart"></canvas></div></div>
    </div>
  </div>

</div>
<script>
const MONTHS={months_json};
let idx=MONTHS.length-1;
let tC,fC,eC,aC,nC,sC,atC,afC,aeC,aaC;
const SC={{5:'#1D9E75',4:'#5DCAA5',3:'#EF9F27',2:'#F09595',1:'#E24B4A'}};

function switchTab(name,btn){{
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
  if(name==='aggregate')updateAggregate();
}}

function fmt(n,d=1){{return(+n||0).toFixed(d)}}
function pct(n){{return fmt(n,1)+'%'}}

// ── Star bars — absolute counts ──────────────────────────────────────────
function renderStarBars(containerId, ratings, totalRated, subId, subText){{
  if(subId) document.getElementById(subId).textContent = subText;
  // Only show stars that have data
  const starsToShow = [5,4,3,2,1].filter(s => (ratings[s]||0) > 0);
  if(!starsToShow.length){{
    document.getElementById(containerId).innerHTML='<div class="empty-state">No rating data</div>';
    return;
  }}
  const mx = Math.max(...starsToShow.map(s => ratings[s]), 1);
  document.getElementById(containerId).innerHTML = starsToShow.map(s=>{{
    const count = ratings[s]||0;
    const barW  = Math.round(count/mx*100);
    return `<div class="star-row">
      <div class="star-label">★${{s}}</div>
      <div class="star-track"><div class="star-fill" style="width:${{barW}}%;background:${{SC[s]}}"></div></div>
      <div class="star-count">${{count.toLocaleString()}}</div>
    </div>`;
  }}).join('');
}}

// ── Monthly tab ──────────────────────────────────────────────────────────
function buildPills(){{
  const row=document.getElementById('pills');
  if(!MONTHS.length){{row.innerHTML='<div class="empty-state">No data yet — run the pipeline first.</div>';return}}
  MONTHS.forEach((m,i)=>{{
    const p=document.createElement('button');
    p.className='pill'+(i===idx?' active':'');p.textContent=m.label;
    p.onclick=()=>{{idx=i;updateAll();row.querySelectorAll('.pill').forEach((el,j)=>el.classList.toggle('active',j===i))}};
    row.appendChild(p);
  }});
}}

function updateMetrics(){{
  const m=MONTHS[idx],prev=idx>0?MONTHS[idx-1]:null;
  document.getElementById('headerMeta').textContent=m.label+' · '+(m.total||0).toLocaleString()+' responses';
  document.getElementById('mTotal').textContent=(m.total||0).toLocaleString();
  document.getElementById('mTotalSub').textContent=(m.totalRated||0).toLocaleString()+' rated responses';
  document.getElementById('mRating').textContent=fmt(m.avgRating);
  document.getElementById('mRatingSub').textContent=prev?(m.avgRating>=prev.avgRating?'↑':'↓')+' vs '+prev.label:'first month';
  document.getElementById('mNsat').textContent=fmt(m.nsat);
  document.getElementById('mNsatSub').textContent=prev?(m.nsat>=prev.nsat?'↑':'↓')+' '+fmt(m.nsat-prev.nsat)+' vs '+prev.label:'first month';
  const fp=m.total>0?Math.round((m.feedback||0)/m.total*100):0;
  document.getElementById('mFeedback').textContent=(m.feedback||0).toLocaleString();
  document.getElementById('mFeedbackPct').textContent=fp+'% of respondents';
}}

function updateRAG(){{
  const m=MONTHS[idx],neg=m.neg||0,neu=m.neu||0,pos=m.pos||0;
  document.getElementById('ragNegLabel').textContent='Negative '+pct(neg);
  document.getElementById('ragNeuLabel').textContent='Neutral '+pct(neu);
  document.getElementById('ragPosLabel').textContent='Positive '+pct(pos);
  document.getElementById('ragBar').innerHTML=
    `<div class="rag-seg" style="flex:${{Math.max(neg,1)}};background:#E24B4A;color:#FCEBEB">${{pct(neg)}}</div>`+
    `<div class="rag-seg" style="flex:${{Math.max(neu,1)}};background:#EF9F27;color:#412402">${{pct(neu)}}</div>`+
    `<div class="rag-seg" style="flex:${{Math.max(pos,1)}};background:#1D9E75;color:#04342C">${{pct(pos)}}</div>`;
}}

function updateStarBars(){{
  const m=MONTHS[idx];
  renderStarBars('starBars', m.ratings, m.totalRated,
    'ratingDistSub', m.label+' · '+(m.totalRated||0).toLocaleString()+' rated responses');
}}

function updateTagMentions(){{
  const tags=(MONTHS[idx].tags||[]).slice().sort((a,b)=>b.n-a.n).slice(0,12);
  if(!tags.length){{document.getElementById('tagMentions').innerHTML='<div class="empty-state">No tag data</div>';return}}
  const mx=Math.max(...tags.map(t=>t.n),1);
  document.getElementById('tagMentions').innerHTML=tags.map(t=>`
    <div class="tag-row">
      <div class="tag-label" title="${{t.t}}">${{t.t}}</div>
      <div class="tag-track"><div class="tag-bar" style="width:${{Math.round(t.n/mx*100)}}%;background:${{t.s>=0?'#1D9E75':'#E24B4A'}}"></div></div>
      <div class="tag-count">${{t.n}}</div>
    </div>`).join('');
}}

function updateDynamicCharts(){{
  const m=MONTHS[idx];
  const tags=(m.tags||[]).slice().sort((a,b)=>a.s-b.s);
  document.getElementById('tagSentWrap').style.height=Math.max(tags.length*32+60,200)+'px';
  if(tC)tC.destroy();
  if(tags.length)tC=new Chart(document.getElementById('tagSentChart'),{{type:'bar',data:{{labels:tags.map(t=>t.t),datasets:[{{label:'Sentiment',data:tags.map(t=>t.s),backgroundColor:tags.map(t=>t.s>=0?'#1D9E7555':'#E24B4A55'),borderColor:tags.map(t=>t.s>=0?'#1D9E75':'#E24B4A'),borderWidth:1}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},indexAxis:'y',scales:{{x:{{min:-1,max:1,ticks:{{callback:v=>v.toFixed(1)}}}},y:{{ticks:{{font:{{size:11}}}}}}}}}}}});
  const feats=(m.features||[]).slice(0,10);
  document.getElementById('featWrap').style.height=Math.max(feats.length*34+60,200)+'px';
  if(fC)fC.destroy();
  if(feats.length)fC=new Chart(document.getElementById('featChart'),{{type:'bar',data:{{labels:feats.map(f=>f.f),datasets:[{{label:'Negative %',data:feats.map(f=>f.neg_pct||0),backgroundColor:'#E24B4A55',borderColor:'#E24B4A',borderWidth:1}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},indexAxis:'y',scales:{{x:{{min:0,max:100,ticks:{{callback:v=>v+'%'}}}},y:{{ticks:{{font:{{size:11}}}}}}}}}}}});
  const errs=(m.errors||[]).slice(0,8);
  document.getElementById('errorWrap').style.height=Math.max(errs.length*34+60,200)+'px';
  if(eC)eC.destroy();
  if(errs.length)eC=new Chart(document.getElementById('errorChart'),{{type:'bar',data:{{labels:errs.map(e=>e.e),datasets:[{{label:'Mentions',data:errs.map(e=>e.n||0),backgroundColor:'#EF9F2755',borderColor:'#EF9F27',borderWidth:1}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},indexAxis:'y',scales:{{y:{{ticks:{{font:{{size:11}}}}}}}}}}}});
}}

function buildStaticCharts(){{
  const labels=MONTHS.map(m=>m.label);
  const opts={{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{autoSkip:false,maxRotation:45}}}}}}}};
  aC=new Chart(document.getElementById('avgRatingChart'),{{type:'line',data:{{labels,datasets:[{{label:'Avg rating',data:MONTHS.map(m=>m.avgRating||0),borderColor:'#BA7517',backgroundColor:'#FAEEDA55',fill:true,tension:0.4,pointRadius:5,pointBackgroundColor:'#BA7517'}}]}},options:{{...opts,scales:{{y:{{min:1,max:5,ticks:{{stepSize:1}}}},x:{{ticks:{{autoSkip:false,maxRotation:45}}}}}}}}}});
  nC=new Chart(document.getElementById('nsatChart'),{{type:'line',data:{{labels,datasets:[{{label:'NSAT',data:MONTHS.map(m=>m.nsat||0),borderColor:'#1D9E75',backgroundColor:'#E1F5EE55',fill:true,tension:0.4,pointRadius:5,pointBackgroundColor:'#1D9E75'}}]}},options:{{...opts,scales:{{y:{{min:-100,max:100}},x:{{ticks:{{autoSkip:false,maxRotation:45}}}}}}}}}});
  sC=new Chart(document.getElementById('sentChart'),{{type:'bar',data:{{labels,datasets:[{{label:'Negative',data:MONTHS.map(m=>m.neg||0),backgroundColor:'#E24B4A',stack:'s'}},{{label:'Neutral',data:MONTHS.map(m=>m.neu||0),backgroundColor:'#EF9F27',stack:'s'}},{{label:'Positive',data:MONTHS.map(m=>m.pos||0),backgroundColor:'#1D9E75',stack:'s'}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:true,position:'bottom',labels:{{font:{{size:11}},boxWidth:10,padding:8}}}}}},scales:{{y:{{stacked:true,max:100,ticks:{{callback:v=>v+'%'}}}},x:{{stacked:true,ticks:{{autoSkip:false,maxRotation:45}}}}}}}}}});
}}

function updateAll(){{updateMetrics();updateRAG();updateStarBars();updateTagMentions();updateDynamicCharts()}}

// ── Aggregate tab ────────────────────────────────────────────────────────
let selectedMonths=new Set(MONTHS.map(m=>m.run_date));

function buildAggPills(){{
  const grid=document.getElementById('aggPills');
  MONTHS.forEach(m=>{{
    const p=document.createElement('button');
    p.className='agg-pill selected';
    p.textContent=m.label;
    p.dataset.runDate=m.run_date;
    p.onclick=()=>{{
      if(selectedMonths.has(m.run_date)){{selectedMonths.delete(m.run_date);p.classList.remove('selected');}}
      else{{selectedMonths.add(m.run_date);p.classList.add('selected');}}
    }};
    grid.appendChild(p);
  }});
}}

function aggSelectAll(){{
  selectedMonths=new Set(MONTHS.map(m=>m.run_date));
  document.querySelectorAll('.agg-pill').forEach(p=>p.classList.add('selected'));
}}
function aggClearAll(){{
  selectedMonths.clear();
  document.querySelectorAll('.agg-pill').forEach(p=>p.classList.remove('selected'));
}}

function updateAggregate(){{
  const sel=MONTHS.filter(m=>selectedMonths.has(m.run_date));
  if(!sel.length){{
    document.getElementById('aTotal').textContent='—';
    document.getElementById('aTotalSub').textContent='No months selected';
    return;
  }}

  // ── Totals ────────────────────────────────────────────────────────────
  const totalResp    =sel.reduce((s,m)=>s+(m.total||0),0);
  const totalFeedback=sel.reduce((s,m)=>s+(m.feedback||0),0);

  // Weighted average rating
  let ratingSum=0,ratingCount=0;
  sel.forEach(m=>{{if(m.avgRating>0){{ratingSum+=m.avgRating*m.total;ratingCount+=m.total;}}}});
  const avgRating=ratingCount>0?ratingSum/ratingCount:0;

  // Aggregate sentiment from raw pct * total
  let posCount=0,negCount=0,neuCount=0;
  sel.forEach(m=>{{
    const t=m.total||0;
    posCount+=Math.round((m.pos||0)/100*t);
    negCount+=Math.round((m.neg||0)/100*t);
    neuCount+=Math.round((m.neu||0)/100*t);
  }});
  const sentTotal=posCount+negCount+neuCount||1;
  const posPct=posCount/sentTotal*100;
  const negPct=negCount/sentTotal*100;
  const neuPct=100-posPct-negPct;

  // ── Aggregate star counts (absolute) ──────────────────────────────────
  const aggRatings={{1:0,2:0,3:0,4:0,5:0}};
  sel.forEach(m=>{{[1,2,3,4,5].forEach(s=>{{aggRatings[s]+=(m.ratings[s]||0);}});}});
  const aggTotalRated=Object.values(aggRatings).reduce((a,b)=>a+b,0);

  // ── Tag aggregation ───────────────────────────────────────────────────
  const tagMap={{}};
  sel.forEach(m=>{{
    (m.tags||[]).forEach(t=>{{
      if(!tagMap[t.t])tagMap[t.t]={{n:0,scoreSum:0}};
      tagMap[t.t].n+=t.n;
      tagMap[t.t].scoreSum+=t.s*t.n;
    }});
  }});
  const aggTags=Object.entries(tagMap)
    .map(([name,v])=>({{t:name,n:v.n,s:v.n>0?v.scoreSum/v.n:0}}))
    .sort((a,b)=>b.n-a.n);

  // ── Feature aggregation ───────────────────────────────────────────────
  const featMap={{}};
  sel.forEach(m=>{{
    (m.features||[]).forEach(f=>{{
      if(!featMap[f.f])featMap[f.f]={{total:0,neg:0}};
      featMap[f.f].total+=f.n;
      featMap[f.f].neg+=Math.round(f.neg_pct/100*f.n);
    }});
  }});
  const aggFeats=Object.entries(featMap)
    .map(([name,v])=>({{f:name,n:v.total,neg_pct:v.total>0?v.neg/v.total*100:0}}))
    .sort((a,b)=>b.neg_pct-a.neg_pct).slice(0,10);

  // ── Error aggregation ─────────────────────────────────────────────────
  const errMap={{}};
  sel.forEach(m=>{{(m.errors||[]).forEach(e=>{{errMap[e.e]=(errMap[e.e]||0)+e.n;}});}});
  const aggErrors=Object.entries(errMap)
    .map(([e,n])=>({{e,n}})).sort((a,b)=>b.n-a.n).slice(0,8);

  // ── Render metrics ────────────────────────────────────────────────────
  document.getElementById('aTotal').textContent=totalResp.toLocaleString();
  document.getElementById('aTotalSub').textContent=sel.length+' month'+(sel.length>1?'s':'')+' selected';
  document.getElementById('aRating').textContent=fmt(avgRating);
  document.getElementById('aPos').textContent=fmt(posPct)+'%';
  document.getElementById('aPosSub').textContent=posCount.toLocaleString()+' positive comments';
  document.getElementById('aNeg').textContent=fmt(negPct)+'%';
  document.getElementById('aNegSub').textContent=negCount.toLocaleString()+' negative comments';

  // ── RAG ───────────────────────────────────────────────────────────────
  document.getElementById('aRagNegLabel').textContent='Negative '+pct(negPct);
  document.getElementById('aRagNeuLabel').textContent='Neutral '+pct(neuPct);
  document.getElementById('aRagPosLabel').textContent='Positive '+pct(posPct);
  document.getElementById('aRagBar').innerHTML=
    `<div class="rag-seg" style="flex:${{Math.max(negPct,1)}};background:#E24B4A;color:#FCEBEB">${{pct(negPct)}}</div>`+
    `<div class="rag-seg" style="flex:${{Math.max(neuPct,1)}};background:#EF9F27;color:#412402">${{pct(neuPct)}}</div>`+
    `<div class="rag-seg" style="flex:${{Math.max(posPct,1)}};background:#1D9E75;color:#04342C">${{pct(posPct)}}</div>`;

  // ── Star bars — absolute counts ───────────────────────────────────────
  renderStarBars('aStarBars', aggRatings, aggTotalRated,
    'aRatingDistSub', sel.map(m=>m.label).join(', '));

  // ── Rating trend chart ────────────────────────────────────────────────
  const allLabels  =MONTHS.map(m=>m.label);
  const allRatings =MONTHS.map(m=>m.avgRating||0);
  const ptColors   =MONTHS.map(m=>selectedMonths.has(m.run_date)?'#BA7517':'#e4e6ea');
  const ptRadius   =MONTHS.map(m=>selectedMonths.has(m.run_date)?6:3);
  if(aaC)aaC.destroy();
  aaC=new Chart(document.getElementById('aAvgRatingChart'),{{
    type:'line',
    data:{{labels:allLabels,datasets:[{{label:'Avg rating',data:allRatings,
      borderColor:'#BA7517',backgroundColor:'#FAEEDA55',fill:true,tension:0.4,
      pointRadius:ptRadius,pointBackgroundColor:ptColors}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
      scales:{{y:{{min:1,max:5,ticks:{{stepSize:1}}}},x:{{ticks:{{autoSkip:false,maxRotation:45}}}}}}}}
  }});

  // ── Tag mentions ──────────────────────────────────────────────────────
  const topTags=aggTags.slice(0,12);
  const mx=Math.max(...topTags.map(t=>t.n),1);
  document.getElementById('aTagMentions').innerHTML=topTags.length
    ?topTags.map(t=>`
      <div class="tag-row">
        <div class="tag-label" title="${{t.t}}">${{t.t}}</div>
        <div class="tag-track"><div class="tag-bar" style="width:${{Math.round(t.n/mx*100)}}%;background:${{t.s>=0?'#1D9E75':'#E24B4A'}}"></div></div>
        <div class="tag-count">${{t.n}}</div>
      </div>`).join('')
    :'<div class="empty-state">No tag data for selected months</div>';

  // ── Tag sentiment chart ───────────────────────────────────────────────
  const sortedTags=[...aggTags].sort((a,b)=>a.s-b.s);
  document.getElementById('aTagSentWrap').style.height=Math.max(sortedTags.length*32+60,200)+'px';
  if(atC)atC.destroy();
  if(sortedTags.length)atC=new Chart(document.getElementById('aTagSentChart'),{{
    type:'bar',
    data:{{labels:sortedTags.map(t=>t.t),datasets:[{{label:'Sentiment',data:sortedTags.map(t=>t.s),
      backgroundColor:sortedTags.map(t=>t.s>=0?'#1D9E7555':'#E24B4A55'),
      borderColor:sortedTags.map(t=>t.s>=0?'#1D9E75':'#E24B4A'),borderWidth:1}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
      indexAxis:'y',scales:{{x:{{min:-1,max:1,ticks:{{callback:v=>v.toFixed(1)}}}},y:{{ticks:{{font:{{size:11}}}}}}}}}}
  }});

  // ── Feature chart ─────────────────────────────────────────────────────
  document.getElementById('aFeatWrap').style.height=Math.max(aggFeats.length*34+60,200)+'px';
  if(afC)afC.destroy();
  if(aggFeats.length)afC=new Chart(document.getElementById('aFeatChart'),{{
    type:'bar',
    data:{{labels:aggFeats.map(f=>f.f),datasets:[{{label:'Negative %',data:aggFeats.map(f=>f.neg_pct||0),
      backgroundColor:'#E24B4A55',borderColor:'#E24B4A',borderWidth:1}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
      indexAxis:'y',scales:{{x:{{min:0,max:100,ticks:{{callback:v=>v+'%'}}}},y:{{ticks:{{font:{{size:11}}}}}}}}}}
  }});

  // ── Error chart ───────────────────────────────────────────────────────
  document.getElementById('aErrorWrap').style.height=Math.max(aggErrors.length*34+60,200)+'px';
  if(aeC)aeC.destroy();
  if(aggErrors.length)aeC=new Chart(document.getElementById('aErrorChart'),{{
    type:'bar',
    data:{{labels:aggErrors.map(e=>e.e),datasets:[{{label:'Mentions',data:aggErrors.map(e=>e.n||0),
      backgroundColor:'#EF9F2755',borderColor:'#EF9F27',borderWidth:1}}]}},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
      indexAxis:'y',scales:{{y:{{ticks:{{font:{{size:11}}}}}}}}}}
  }});
}}

// ── Init ─────────────────────────────────────────────────────────────────
if(MONTHS.length){{
  buildPills();
  updateAll();
  buildStaticCharts();
  buildAggPills();
}}else{{
  document.getElementById('pills').innerHTML=
    '<div class="empty-state">No pipeline data found. Run the pipeline first.</div>';
}}
</script>
</body>
</html>"""

def generate():
    result = load_data()
    if result is None: return
    summary, tags, features, errors = result
    if summary.empty:
        print("No data in monthly_summary yet. Run the pipeline first.")
        return
    print(f"Loaded {len(summary)} months of data")
    months = build_months_js(summary, tags, features, errors)
    print(f"Months: {[m['label'] for m in months]}")
    html = build_dashboard(months)
    os.makedirs(os.path.dirname(OUTPUT_LATEST), exist_ok=True)
    with open(OUTPUT_LATEST, "w", encoding="utf-8") as f: f.write(html)
    print(f"Saved: {OUTPUT_LATEST}")
    os.makedirs(os.path.dirname(OUTPUT_ARCHIVE), exist_ok=True)
    with open(OUTPUT_ARCHIVE, "w", encoding="utf-8") as f: f.write(html)
    print(f"Archived: {OUTPUT_ARCHIVE}")

if __name__ == "__main__":
    generate()