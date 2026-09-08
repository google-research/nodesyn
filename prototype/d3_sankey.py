"""D3-based Sankey diagram with HTML-rendered labels for consistent fonts."""
import json
import pandas as pd
import textwrap


def create_d3_sankey_html(df_final, height=700, scale_factor=1.0):
    """Generate a self-contained HTML Sankey using D3 + d3-sankey.

    Labels use foreignObject so they render as HTML (matching page fonts).
    Filters are native HTML <select> elements with JS-driven re-rendering.
    """
    if df_final.empty:
        return "<p>No data</p>"

    # --- Data processing (mirrors the original Plotly pipeline) ---
    df_temp = df_final.loc[:, ~df_final.columns.duplicated()].copy()

    def safe_eval_list(x):
        if isinstance(x, str) and x.strip().startswith('['):
            try:
                return eval(x)
            except Exception:
                return [x]
        return x if isinstance(x, list) else [x]

    for col in ['level3', 'extracted_Country', 'user_group']:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].apply(safe_eval_list)

    df_exploded = df_temp.copy()
    for col in ['level3', 'extracted_Country', 'user_group']:
        if col in df_exploded.columns:
            df_exploded = df_exploded.explode(col)
    df_exploded = df_exploded.reset_index(drop=True)

    if 'extracted_Country' in df_exploded.columns and 'cleaned_Country' not in df_exploded.columns:
        df_exploded.rename(columns={'extracted_Country': 'cleaned_Country'}, inplace=True)
    elif 'extracted_Country' in df_exploded.columns and 'cleaned_Country' in df_exploded.columns:
        df_exploded.drop(columns=['extracted_Country'], inplace=True, errors='ignore')

    df_exploded = df_exploded.loc[:, ~df_exploded.columns.duplicated()]

    flow_cols = ['Domain', 'level1', 'level2', 'level3', 'user_group', 'cleaned_Country']
    flow_cols = [c for c in flow_cols if c in df_exploded.columns]

    for col in flow_cols:
        series_col = df_exploded[col]
        if isinstance(series_col, pd.DataFrame):
            series_col = series_col.iloc[:, 0]
        df_exploded[col] = series_col.astype(str).str.replace('-', '', regex=False).str.strip()
        df_exploded[col] = df_exploded[col].replace({
            'UK': 'United Kingdom', 'USA': 'United States',
            'US': 'United States', 'America': 'United States'
        })

    # Filter config
    filter_map = {
        'user_group': 'User Groups', 'level1': 'Level 1s',
        'user_case': 'User Cases', 'model_modality': 'Model Modalities',
        'cleaned_Country': 'Countries',
    }
    filters = []
    for col, label in filter_map.items():
        if col in df_exploded.columns:
            opts = sorted(df_exploded[col].dropna().unique().tolist())
            if opts:
                filters.append({'column': col, 'label': label, 'options': opts})

    # Send only needed columns as JSON
    keep = list(set(flow_cols + [f['column'] for f in filters]))
    keep = [c for c in keep if c in df_exploded.columns]
    data_json = df_exploded[keep].to_json(orient='records')

    colors = {
        'Domain': '#4f46e5', 'level1': '#7c3aed', 'level2': '#db2777',
        'level3': '#ea580c', 'user_group': '#059669', 'cleaned_Country': '#0284c7',
    }
    light_colors = {
        'Domain': 'rgba(79,70,229,0.22)', 'level1': 'rgba(124,58,237,0.20)',
        'level2': 'rgba(219,39,119,0.18)', 'level3': 'rgba(234,88,12,0.18)',
        'user_group': 'rgba(5,150,105,0.18)', 'cleaned_Country': 'rgba(2,132,199,0.18)',
    }

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12.3"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif}}
body{{background:white}}
.filters{{display:flex;gap:10px;padding:8px 4px 12px;flex-wrap:wrap;align-items:center}}
.filter-select{{font-family:'Inter',sans-serif;font-size:12px;font-weight:500;color:#334155;
  padding:6px 26px 6px 10px;border:1px solid #e2e8f0;border-radius:8px;background:#fff;
  cursor:pointer;appearance:none;-webkit-appearance:none;min-width:120px;
  background-image:url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2394a3b8' d='M3 5l3 3 3-3'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center}}
.filter-select:hover{{border-color:#cbd5e1}}
.filter-select:focus{{outline:none;border-color:#6366f1;box-shadow:0 0 0 3px rgba(99,102,241,0.1)}}
.link{{fill:none;transition:stroke-opacity 0.2s}}
.link:hover{{stroke-opacity:0.45}}
.node-label{{font-size:11px;font-weight:600;color:#334155;line-height:1.25;overflow:hidden;
  text-overflow:ellipsis;pointer-events:none;word-wrap:break-word}}
.tooltip{{position:fixed;background:#0f172a;color:white;padding:8px 12px;border-radius:8px;
  font-size:12px;font-weight:500;pointer-events:none;opacity:0;transition:opacity 0.15s;
  z-index:100;box-shadow:0 4px 12px rgba(0,0,0,0.25);max-width:260px;line-height:1.4}}
.empty-msg{{padding:60px 20px;color:#94a3b8;text-align:center;font-size:14px}}
</style></head><body>
<div class="filters" id="filters"></div>
<div id="chart"></div>
<div class="tooltip" id="tooltip"></div>
<script>
const rawData={data_json};
const flowCols={json.dumps(flow_cols)};
const levelColors={json.dumps(colors)};
const lightColors={json.dumps(light_colors)};
const filterCfg={json.dumps(filters)};
const levelLabels={json.dumps({c: c.replace('cleaned_Country','Country').replace('level','L').replace('user_group','User Group').replace('Domain','Domain') for c in flow_cols})};
const scale = {scale_factor};
const W = Math.max(document.documentElement.clientWidth - 20, 800) * scale;
const H = {height} * scale;
const mg={{top:8,right:150,bottom:8,left:8}};
let activeFilters={{}};

/* Build filter dropdowns */
const fd=document.getElementById('filters');
filterCfg.forEach(f=>{{
  const s=document.createElement('select');s.className='filter-select';s.dataset.col=f.column;
  const a=document.createElement('option');a.value='__all__';a.textContent='All '+f.label;s.appendChild(a);
  f.options.forEach(o=>{{const op=document.createElement('option');op.value=o;op.textContent=o;s.appendChild(op)}});
  s.addEventListener('change',()=>{{
    if(s.value==='__all__')delete activeFilters[f.column];else activeFilters[f.column]=s.value;
    render()}});
  fd.appendChild(s)}});

function genFlow(data){{
  const pairs=[];
  for(let i=0;i<flowCols.length-1;i++){{
    data.forEach(r=>{{
      const src=String(r[flowCols[i]]||'').trim();
      const tgt=String(r[flowCols[i+1]]||'').trim();
      if(src&&tgt&&src!==tgt)pairs.push({{s:src,t:tgt,sc:flowCols[i],tc:flowCols[i+1]}})}})}};
  const ct={{}};
  pairs.forEach(p=>{{const k=p.s+'|||'+p.t;if(!ct[k])ct[k]={{s:p.s,t:p.t,v:0,sc:p.sc,tc:p.tc}};ct[k].v++}});
  const links=Object.values(ct).filter(l=>l.v>0);
  /* node→level from data columns (last column wins, then earlier overrides) */
  const nm={{}};
  [...flowCols].reverse().forEach(col=>{{data.forEach(r=>{{const v=String(r[col]||'').trim();if(v)nm[v]=col}})}});
  const names=[...new Set(links.flatMap(l=>[l.s,l.t]))].sort();
  const nodes=names.map(n=>({{name:n,level:nm[n]||'Domain'}}));
  const ni={{}};nodes.forEach((n,i)=>ni[n.name]=i);
  const iLinks=links.map(l=>({{source:ni[l.s],target:ni[l.t],value:l.v}}))
    .filter(l=>l.source!==undefined&&l.target!==undefined&&l.source!==l.target);
  return{{nodes,links:iLinks}}}};

function render(){{
  let d=rawData;
  Object.entries(activeFilters).forEach(([c,v])=>{{d=d.filter(r=>String(r[c])===v)}});
  const chart=document.getElementById('chart');
  if(!d.length){{chart.innerHTML='<p class="empty-msg">No data matches filters.</p>';return}};
  const fd=genFlow(d);
  if(!fd.nodes.length){{chart.innerHTML='<p class="empty-msg">No flow data.</p>';return}};
  chart.innerHTML='';
  const svg=d3.select('#chart').append('svg').attr('width',W).attr('height',H);
  const sankey=d3.sankey().nodeId(d=>d.index).nodeWidth(24).nodePadding(16)
    .nodeAlign(d3.sankeyJustify).extent([[mg.left,mg.top],[W-mg.right,H-mg.bottom]]);
  const graph=sankey({{nodes:fd.nodes.map(d=>({{...d}})),links:fd.links.map(d=>({{...d}}))}});
  const tip=document.getElementById('tooltip');

  /* Links */
  const linkG=svg.append('g');
  const linkPaths=linkG.selectAll('.link').data(graph.links).join('path')
    .attr('class','link').attr('d',d3.sankeyLinkHorizontal())
    .attr('stroke',d=>lightColors[d.source.level]||'rgba(148,163,184,0.2)')
    .attr('stroke-width',d=>Math.max(1,d.width))
    .attr('stroke-opacity',0.7)
    .on('mouseover',(e,d)=>{{tip.style.opacity=1;
      tip.innerHTML='<b>'+d.source.name+'</b> → <b>'+d.target.name+'</b><br>'+d.value+' connections'}})
    .on('mousemove',e=>{{tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px'}})
    .on('mouseout',()=>{{tip.style.opacity=0}});

  /* Nodes — each <g> is translated to (x0, y0); children use local coords */
  const ng=svg.append('g').selectAll('.node').data(graph.nodes).join('g')
    .attr('class','node')
    .attr('transform',d=>`translate(${{d.x0}},${{d.y0}})`);

  ng.append('rect')
    .attr('x',0).attr('y',0)
    .attr('height',d=>Math.max(2,d.y1-d.y0)).attr('width',d=>d.x1-d.x0)
    .attr('fill',d=>levelColors[d.level]||'#94a3b8').attr('rx',3).attr('ry',3)
    .style('cursor','grab')
    .on('mouseover',(e,d)=>{{tip.style.opacity=1;
      tip.innerHTML='<b>'+d.name+'</b><br><span style="color:#94a3b8">'+
        (d.level==='cleaned_Country'?'Country':d.level.replace('level','L').replace('user_group','User Group'))+'</span><br>'+d.value+' total'}})
    .on('mousemove',e=>{{tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px'}})
    .on('mouseout',()=>{{tip.style.opacity=0}});

  /* Labels via foreignObject — positioned relative to group */
  const labels=ng.append('foreignObject')
    .attr('x',d=>d.x0<W/2?d.x1-d.x0+6:-136)
    .attr('y',d=>(d.y1-d.y0)/2-18)
    .attr('width',130).attr('height',40);
  labels.append('xhtml:div').attr('class','node-label')
    .style('text-align',d=>d.x0<W/2?'left':'right')
    .style('display','flex').style('align-items','center')
    .style('height','100%')
    .style('justify-content',d=>d.x0<W/2?'flex-start':'flex-end')
    .text(d=>d.name);

  /* Helper: move a node group to its current y0 position */
  function applyTransform(sel, animate){{
    sel.each(function(d){{
      const el=d3.select(this);
      if(animate){{
        el.style('transition','transform 160ms ease-out');
      }} else {{
        el.style('transition','none');
      }}
      el.attr('transform',`translate(${{d.x0}},${{d.y0}})`);
    }});
  }}

  ng.call(d3.drag()
    .subject(function(event,d){{return {{x:d.x0,y:d.y0}}}})
    .on('start',function(event,d){{
      d3.select(this).raise();
      d3.select(this).select('rect').style('cursor','grabbing')
        .attr('filter','drop-shadow(0 4px 10px rgba(0,0,0,0.2))');
    }})
    .on('drag',function(event,d){{
      const h = d.y1 - d.y0;
      const w = d.x1 - d.x0;
      
      d.x0 = Math.max(mg.left, Math.min(W - mg.right - w, event.x));
      d.x1 = d.x0 + w;
      
      d.y0 = Math.max(mg.top, Math.min(H - mg.bottom - h, event.y));
      d.y1 = d.y0 + h;
      
      applyTransform(d3.select(this), false);
      
      try {{ sankey.update(graph); }} catch(e) {{ console.error(e); }}
      linkPaths.attr('d',d3.sankeyLinkHorizontal());
    }})
    .on('end',function(event,d){{
      d3.select(this).select('rect').style('cursor','grab').attr('filter',null);
    }}));


}}
render();
</script></body></html>"""
    return html
