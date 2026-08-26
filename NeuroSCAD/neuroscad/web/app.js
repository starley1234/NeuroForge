const $ = (id) => document.getElementById(id);
const state = { ir: null, scad: '', overrides: {}, tab: 'scad', timer: null };

async function api(path, body) {
  const response = await fetch(path, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
  if (!response.ok) { let msg=`HTTP ${response.status}`; try { const x=await response.json(); msg=x.detail||msg; } catch {} throw new Error(msg); }
  return response;
}
function setBusy(value) { $('generate').disabled=value; $('generate').querySelector('span').textContent=value?'Компиляция…':'Скомпилировать намерение'; }
function showCode() { $('code').textContent = state.tab === 'scad' ? state.scad : JSON.stringify(state.ir, null, 2); }
function defaults() { return Object.fromEntries((state.ir?.parameters||[]).map(p=>[p.name,p.default])); }
function selected() { return {...defaults(), ...state.overrides}; }

function renderDiagram() {
  if (!state.ir) return; const p=selected(), family=state.ir.metadata?.family||'split_pipe_clamp';
  const visible=(id,on)=>$(id).style.display=on?'':'none';
  ['outer','inner','gap','lug','bolt'].forEach(id=>visible(id,false)); visible('plateHoles',false);
  if (family==='mounting_plate') {
    const scale=Math.min(3,280/p.plate_length,160/p.plate_width), w=p.plate_length*scale, h=p.plate_width*scale;
    visible('lug',true); visible('plateHoles',true); $('lug').setAttribute('x',-w/2); $('lug').setAttribute('y',-h/2); $('lug').setAttribute('width',w); $('lug').setAttribute('height',h);
    const dx=(p.plate_length/2-p.edge_offset)*scale, dy=(p.plate_width/2-p.edge_offset)*scale, r=p.hole_d*scale/2;
    [...$('plateHoles').children].forEach((hole,i)=>{hole.setAttribute('cx',(i<2?1:-1)*dx);hole.setAttribute('cy',(i%2?1:-1)*dy);hole.setAttribute('r',r);});
    $('diagramLabel').textContent=`PLATE / ${p.plate_length} × ${p.plate_width} × ${p.thickness}`;
  } else if (family==='bushing') {
    const scale=Math.min(4,220/p.outer_d); visible('outer',true); visible('inner',true); $('outer').setAttribute('r',p.outer_d*scale/2); $('inner').setAttribute('r',p.inner_d*scale/2);
    $('diagramLabel').textContent=`BUSHING / Ø${p.outer_d} / BORE Ø${p.inner_d} / H ${p.height}`;
  } else {
    const outerD=(p.tube_d||25)+2*(p.wall||4), scale=Math.min(3.2,220/outerD), outer=outerD*scale/2, inner=(p.tube_d||25)*scale/2;
    ['outer','inner','gap','lug','bolt'].forEach(id=>visible(id,true)); $('outer').setAttribute('r',outer); $('inner').setAttribute('r',inner);
    const x=outer-2, length=p.lug_length*scale, h=p.lug_height*scale; $('lug').setAttribute('x',x); $('lug').setAttribute('y',-h/2); $('lug').setAttribute('width',length); $('lug').setAttribute('height',h);
    $('gap').setAttribute('x',inner-2); $('gap').setAttribute('y',-p.split_gap*scale/2); $('gap').setAttribute('width',outer-inner+length+8); $('gap').setAttribute('height',p.split_gap*scale);
    $('bolt').setAttribute('cx',x+length/2); $('bolt').setAttribute('r',p.screw_d*scale/2); $('diagramLabel').textContent=`CLAMP / TUBE Ø${p.tube_d} / WALL ${p.wall}`;
  }
}
function renderParameters() {
  const root=$('parameters'); root.textContent='';
  for (const p of state.ir.parameters) {
    const wrap=document.createElement('div'); wrap.className='param';
    const line=document.createElement('div'); line.className='param-line';
    const label=document.createElement('span'); label.textContent=p.label||p.name;
    const value=document.createElement('span'); value.textContent=`${p.default} ${p.unit}`;
    const input=document.createElement('input'); input.type='range'; input.min=p.minimum; input.max=p.maximum; input.step=p.step; input.value=p.default; input.dataset.name=p.name;
    input.addEventListener('input',()=>{ state.overrides[p.name]=Number(input.value); value.textContent=`${input.value} ${p.unit}`; renderDiagram(); clearTimeout(state.timer); state.timer=setTimeout(recompile,180); });
    line.append(label,value); wrap.append(line,input); root.append(wrap);
  }
}
async function recompile() {
  try { const response=await api('/v1/compile',{ir:state.ir,overrides:state.overrides}); const data=await response.json(); state.scad=data.openscad; setValidation(data.validation); showCode(); }
  catch(e){ $('error').textContent=e.message; }
}
function setValidation(report) { $('level').textContent=report.level; $('validBadge').textContent=report.valid?`VALID / ${report.level.toUpperCase()}`:'INVALID'; $('validBadge').classList.toggle('pass',report.valid); }
async function generate() {
  $('error').textContent=''; setBusy(true);
  try { const response=await api('/v1/generate',{prompt:$('prompt').value,validate_geometry:false}); const data=await response.json(); state.ir=data.ir; state.scad=data.openscad; state.overrides={}; renderParameters(); renderDiagram(); setValidation(data.validation); showCode(); ['downloadScad','downloadIr','downloadStl'].forEach(x=>$(x).disabled=false); }
  catch(e){ $('error').textContent=e.message; } finally { setBusy(false); }
}
function save(blob,name){ const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000); }
async function exportArtifact(kind) {
  if (kind==='ir') return save(new Blob([JSON.stringify(state.ir,null,2)],{type:'application/json'}),'neuroscad.json');
  try { const response=await api(`/v1/export/${kind}`,{ir:state.ir,overrides:state.overrides}); save(await response.blob(),`neuroscad.${kind}`); }
  catch(e){ $('error').textContent=e.message; }
}
$('generate').addEventListener('click',generate);
document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>{$('prompt').value=b.dataset.prompt;generate();}));
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.tab=b.dataset.tab;showCode();}));
$('downloadScad').addEventListener('click',()=>exportArtifact('scad')); $('downloadIr').addEventListener('click',()=>exportArtifact('ir')); $('downloadStl').addEventListener('click',()=>exportArtifact('stl'));
fetch('/health/ready').then(r=>r.json()).then(x=>{$('systemText').textContent=x.openscad?'API + OPENSCAD READY':'API READY / RENDER OFFLINE';}).catch(()=>{$('systemText').textContent='API UNAVAILABLE';});
