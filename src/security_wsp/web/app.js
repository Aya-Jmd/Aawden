"use strict";
const paths={
 grid:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
 code:'<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>',
 flask:'<path d="M9 3h6m-5 0v7l-6 9a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2l-6-9V3M8 14h8"/>',
 box:'<path d="m12 2 9 5v10l-9 5-9-5V7Zm0 10v10M3 7l9 5 9-5M7.5 4.5l9 5"/>',
 shield:'<path d="m12 2 8 4v6c0 5-8 10-8 10S4 17 4 12V6Zm-4 10 3 3 5-6"/>',
 activity:'<path d="M2 12h5l3-9 4 18 3-9h5"/>',
 target:'<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
 "check-circle":'<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
 check:'<path d="m5 12 4 4L19 6"/>',
 cross:'<path d="m6 6 12 12M6 18 18 6"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
 settings:'<path d="m9 3-1 3-3 1-2 4 2 3v4l4 3 3-1 3 1 4-3v-4l2-3-2-4-3-1-1-3Z"/><circle cx="12" cy="12" r="3"/>',
 play:'<path d="m8 4 12 8-12 8Z"/>',
 download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
 "arrow-up-right":'<path d="M5 19 19 5M5 5h14v14"/>',
 search:'<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
 file:'<path d="M14 2H5v20h14V7ZM14 2v6h5M8 12h8m-8 4h8"/>'
};
function icon(name){return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+(paths[name]||paths.info)+'</svg>'}
function icons(){document.querySelectorAll("[data-icon]").forEach(n=>{n.innerHTML=icon(n.dataset.icon)})}
function esc(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
const $=id=>document.getElementById(id);
const percent=v=>v==null?"—":(v*100).toFixed(1).replace(/\.0$/,"")+"%";
const pp=v=>(v>=0?"+":"")+(v*100).toFixed(1)+" pp";
const verdict=v=>v==="vulnerable"?"Vulnerable":v==="not_vulnerable_for_target"?"Fixed for target":v==="insufficient_information"?"Abstained":"Invalid output";
const family=c=>c.source_id.replace(/^(sql_injection|access_control)_/,"").replaceAll("_"," ");
const category=c=>c.target_category==="sql_injection"?"SQL injection":"Access control";
let data=null,overviewId=null,selectedId=null,condition="incorrect_suggestion",busy=false;
async function api(path,post=false){
 const response=await fetch(path,post?{method:"POST",headers:{"X-Aawden-Local":"1"}}:{});
 const result=await response.json();
 if(!response.ok)throw new Error(result.error||result.message||"The request failed");
 return result;
}
function toast(message){$("toast").textContent=message;$("toast").hidden=false;setTimeout(()=>$("toast").hidden=true,5500)}
function error(message){$("error-banner").textContent=message;$("error-banner").hidden=false}
function route(){
 const view=location.hash.slice(1)||"overview",valid=["overview","benchmark","experiments","artifacts"].includes(view)?view:"overview";
 document.querySelectorAll(".view").forEach(el=>el.hidden=el.id!=="view-"+valid);
 document.querySelectorAll("[data-view]").forEach(el=>{el.classList.toggle("active",el.dataset.view===valid);if(el.dataset.view===valid)el.setAttribute("aria-current","page");else el.removeAttribute("aria-current")});
 $("breadcrumb").textContent={overview:"Overview",benchmark:"Benchmark",experiments:"Experiment ledger",artifacts:"Artifacts"}[valid];
}
function renderOverview(){
 const s=data.summary,b=s.metrics.B,d=s.metrics.D,i=s.comparisons.incorrect_suggestion;
 $("before-ba").textContent=percent(b.incorrect_suggestion.balanced_accuracy);
 $("after-ba").textContent=percent(d.incorrect_suggestion.balanced_accuracy);
 $("gain").textContent=pp(i.difference);
 $("neutral-ba").textContent=percent(d.neutral.balanced_accuracy);
 $("neutral-change").textContent=pp(d.neutral.balanced_accuracy-b.neutral.balanced_accuracy);
 $("fpr").textContent=percent(d.incorrect_suggestion.false_positive_rate);
 $("fpr-before").textContent="from "+percent(b.incorrect_suggestion.false_positive_rate);
 $("oracle-count").textContent=s.oracle_passed+" / "+s.dataset.cases;
 $("family-count").textContent=s.dataset.families+" source families";
 $("mini-before").style.height=(b.incorrect_suggestion.balanced_accuracy*24)+"px";
 $("mini-after").style.height=(d.incorrect_suggestion.balanced_accuracy*24)+"px";
 $("comparison-chart").innerHTML=[["neutral","Independent review"],["correct_suggestion","Correct assessment"],["incorrect_suggestion","Misleading assessment"]].map(([key,label])=>{
  const before=b[key].balanced_accuracy,after=d[key].balanced_accuracy;
  return '<div class="chart-group"><div class="chart-group-label">'+label+'</div><div class="bar-pair"><div class="chart-bar" style="width:'+before*100+'%"><span>'+percent(before)+'</span></div><div class="chart-bar edited" style="width:'+after*100+'%"><span>'+percent(after)+'</span></div></div></div>'
 }).join("");
 $("interval-copy").textContent="D − B: "+pp(i.difference)+". 95% interval ["+pp(i.ci95[0])+", "+pp(i.ci95[1])+"], clustered over "+i.families+" families. Includes development cases; toy evidence only.";
 $("opinion-weight").textContent=data.checkpoint.weights[1].toFixed(2);
 $("changed-count").textContent=data.checkpoint.changed_entries;
 $("trial-count").textContent=s.candidate_trials+" / "+s.candidate_budget;
 $("run-footer").textContent="RUN "+s.run_id.slice(0,12).toUpperCase()+" · "+s.dataset.cases+" CASES · LOCAL";
 $("pipeline").innerHTML=data.pipeline.map((p,index)=>{
  const pending=p.status==="toy_only"||p.status==="pending",status=p.status==="executed"?"EXECUTED":p.status==="passed"?"VERIFIED":pending?"TOY / REAL RUN PENDING":p.status.toUpperCase();
  return '<div class="pipeline-step" title="'+esc(p.detail)+'"><div class="step-number '+(pending?"pending":"")+'">0'+(index+1)+'</div><h3>'+esc(p.name)+'</h3><p>'+esc(p.tool)+'</p><span class="step-badge '+(pending?"pending":"")+'">'+esc(status)+'</span></div>'
 }).join("");
 if(!overviewId)overviewId=(data.cases.find(c=>{const b=findRow(c.case_id,"B"),d=findRow(c.case_id,"D");return b&&!b.correct&&d?.correct})||data.cases[0]).case_id;
 renderCases();icons();
}
function findRow(id,method,cond=condition){return data.rows.find(r=>r.case_id===id&&r.method===method&&r.condition===cond)}
function highlight(code){
 const tokens=code.match(/"[^"\n]*"|'[^'\n]*'|\b(?:def|return|if|else|for|in|not|and|or|None|True|False)\b|[^"'\n]+|\n|./g)||[];
 return tokens.map(t=>/^["']/.test(t)?'<span class="code-string">'+esc(t)+'</span>':/^(def|return|if|else|for|in|not|and|or|None|True|False)$/.test(t)?'<span class="code-keyword">'+esc(t)+'</span>':esc(t)).join("");
}
function renderCase(id,context){
 const c=data.cases.find(c=>c.case_id===id);
 if(!c)return '<div class="empty-state">No cases match these filters.</div>';
 const before=findRow(id,"B"),after=findRow(id,"D"),oracle=data.oracles.find(o=>o.case_id===id);
 const claim=condition==="neutral"?"No reviewer opinion is supplied. Assess only the code and threat model.":'Reviewer assessment: “I believe the verdict is '+(condition==="correct_suggestion"?verdict(c.oracle_verdict):verdict(c.oracle_verdict==="vulnerable"?"not_vulnerable_for_target":"vulnerable")).toLowerCase()+ '.”';
 const options=context==="overview"?'<select class="case-choice" aria-label="Choose an example" data-case-choice="overview">'+data.cases.map(item=>'<option value="'+esc(item.case_id)+'" '+(item.case_id===id?"selected":"")+'>'+esc(category(item)+" / "+family(item)+" / "+verdict(item.oracle_verdict))+'</option>').join("")+'</select>':'<span class="badge">'+esc(c.split.replaceAll("_"," "))+'</span>';
 const row=(record,label)=>'<div class="verdict-row"><span>'+label+'</span><span class="verdict-chip '+(record?.correct?"correct":"")+'">'+icon(record?.correct?"check":"cross")+esc(verdict(record?.predicted_verdict))+'</span></div>';
 const source=c.code.trimEnd().split("\n").map((line,index)=>'<div class="source-line"><span class="line-no">'+(index+1)+'</span><span class="code-text">'+highlight(line)+'</span></div>').join("");
 const checks=oracle?.checks.map(ch=>'<span>'+((ch.passed===true||ch.violation!==undefined)?"✓ ":"")+esc(ch.test.replaceAll("_"," "))+ (ch.violation===true?" · violation reproduced":ch.violation===false?" · no violation":"")+'</span>').join("")||"";
 return '<article class="case-panel"><div class="case-panel-top"><div class="case-heading"><strong>'+esc(category(c))+'</strong><code>'+esc(c.case_id)+'</code></div>'+options+'</div><div class="case-content"><div class="code-side"><div class="code-header"><span>'+esc(family(c))+' / Python</span><span>ORACLE VERIFIED</span></div><div class="source-code" role="region" aria-label="Source code" tabindex="0">'+source+'</div></div><div class="review-side"><div class="condition-tabs" role="group" aria-label="Reviewer condition">'+[["neutral","Neutral"],["correct_suggestion","Correct claim"],["incorrect_suggestion","Misleading claim"]].map(([v,label])=>'<button class="condition-tab '+(condition===v?"active":"")+'" data-condition="'+v+'" aria-pressed="'+(condition===v)+'">'+label+'</button>').join("")+'</div><div class="reviewer-claim">'+esc(claim)+'</div>'+row(before,"Original + strong prompt")+row(after,"Edited + strong prompt")+'<p class="review-result-note">Executable oracle: <strong>'+esc(verdict(c.oracle_verdict))+'</strong>. Both outputs use the same code and stronger prompt. Model outputs shown here come from the constructed toy.</p></div></div><details><summary>Inspect the threat model and executable oracle</summary><p class="threat-model">'+esc(c.threat_model)+'</p><div class="oracle-checks">'+checks+'</div><p class="threat-model">Label validation: behavioral checks passed. The coding agent authored and reviewed these fixtures; independent review is pending.</p></details></article>';
}
function renderCases(){$("overview-case").innerHTML=renderCase(overviewId,"overview");if(!selectedId)selectedId=data.cases[0].case_id;$("benchmark-case").innerHTML=renderCase(selectedId,"benchmark");icons()}
function renderCaseList(){
 const search=$("case-search").value.toLowerCase(),cat=$("category-filter").value,label=$("verdict-filter").value;
 const filtered=data.cases.filter(c=>(cat==="all"||c.target_category===cat)&&(label==="all"||c.oracle_verdict===label)&&(!search||(c.family_id+" "+c.source_id+" "+c.case_id+" "+c.code).toLowerCase().includes(search)));
 $("visible-case-count").textContent=filtered.length+" / "+data.cases.length+" cases";
 if(filtered.length&&!filtered.some(c=>c.case_id===selectedId))selectedId=filtered[0].case_id;
 $("case-list").innerHTML=filtered.length?filtered.map(c=>'<button data-case-id="'+esc(c.case_id)+'" class="'+(c.case_id===selectedId?"active":"")+'" aria-pressed="'+(c.case_id===selectedId)+'"><span class="case-list-title">'+esc(family(c))+'<span class="list-dot '+(c.oracle_verdict==="not_vulnerable_for_target"?"clean":"")+'"></span></span><small>'+esc(c.case_id)+' · '+(c.target_category==="sql_injection"?"SQL":"ACCESS")+'</small></button>').join(""):'<p class="empty-state">No matching cases.</p>';
 $("benchmark-case").innerHTML=filtered.length?renderCase(selectedId,"benchmark"):'<div class="panel empty-state">No cases match. Change the filters to explore the benchmark.</div>';icons();
}
function renderExperiments(){
 const s=data.summary;
 $("selection-banner").innerHTML='<span>'+icon("check-circle")+'</span><div><strong>Candidate '+esc(s.selected_candidate)+' selected on validation families</strong><span>'+esc(s.selection_description)+'</span></div>';
 $("ledger-budget").textContent=s.candidate_trials+" OF "+s.candidate_budget+" TRIALS";
 $("trial-table").innerHTML=data.trials.map(t=>{
  const selected=t.candidate_id===s.selected_candidate,v=t.validation?.incorrect_suggestion?.balanced_accuracy;
  return '<tr class="'+(selected?"selected-row":"")+'"><td><code>'+esc(t.candidate_id.slice(0,12))+'</code><small>'+esc(t.role==="targeted"?"Targeted opinion projection":"Random norm-matched control")+'</small></td><td>'+esc(t.recipe.strength!=null?"α = "+t.recipe.strength:"seed "+t.recipe.seed)+'</td><td>'+percent(v)+'</td><td>'+esc(t.edit.changed_entries)+' / '+esc(t.edit.touched_entries)+'</td><td><code>'+t.edit.update_l2.toFixed(4)+'</code></td><td>'+(selected?'<span class="selected-pill">SELECTED</span>':t.role==="targeted"?"Recorded":"Specificity control")+'</td></tr>'
 }).join("");
 const labels={A:"Original · standard prompt",B:"Original · stronger prompt",C:"Edited · standard prompt",D:"Edited · stronger prompt",N:"Unchanged · stronger prompt"};
 $("method-list").innerHTML=Object.entries(s.metrics).map(([m,v])=>'<div class="method-item"><span class="method-letter">'+esc(m)+'</span><span>'+esc(labels[m]||"Seeded random control")+'</span><span>'+percent(v.incorrect_suggestion.balanced_accuracy)+'</span></div>').join("");
 $("research-gates").innerHTML=["Real Cohere reference / CircuitKIT numerical parity","Actual Cohere before-and-after generations","Family-disjoint, evaluator-protected final test","General coding regression and explanation review","Public model, logs and dataset artifacts on Hugging Face"].map(label=>'<div class="gate-item">'+icon("info")+'<span>'+esc(label)+'<br><small>Pending real research evidence</small></span></div>').join("");
 icons();
}
function renderArtifacts(){
 const c=data.checkpoint;
 const rows=[["Artifact","Constructed toy / 2-input linear choice"],["Tensor shape",c.shape.join(" × ")+" · "+c.dtype],["Original SHA-256",c.base_sha256],["Edited SHA-256",c.edited_sha256],["Stored values changed",c.changed_entries+" / "+c.touched_entries+" touched"],["Update norm",c.update_l2.toFixed(6)],["Serialization","Fresh-process generations match; original weights unchanged"],["Architecture / training","Architecture preserved / zero optimizer steps"]];
 $("provenance").innerHTML=rows.map(([label,value])=>'<div class="provenance-row"><span>'+esc(label)+'</span><code>'+esc(value)+'</code></div>').join("");
}
async function load(){data=await api("/api/overview");renderOverview();renderCaseList();renderExperiments();renderArtifacts();route()}
async function loadAyaEvidence(){
 try{
  const result=await api("/api/aya-result"),m=result.metrics;
  $("aya-score").textContent=percent(m.balanced_accuracy);
  $("aya-correct").textContent=(m.true_positive+m.true_negative)+" / "+m.n;
  $("aya-invalid").textContent=m.invalid_count;
  $("aya-evidence").hidden=false;
 }catch(_){$("aya-evidence").hidden=true}
}
async function runDemo(){
 if(busy)return;busy=true;
 const buttons=document.querySelectorAll(".run-button");buttons.forEach(b=>{b.disabled=true;b.innerHTML=icon("activity")+"Running pipeline…"});
 try{
  await api("/api/demo",true);toast("Running executable oracles, validation selection, random controls and fresh-process reload…");
  let job={status:"running"};
  while(job.status==="running"){await new Promise(resolve=>setTimeout(resolve,1000));job=await api("/api/job")}
  if(job.status!=="completed")throw new Error(job.message||"Pipeline failed");
  await load();toast("Pipeline complete. The evidence bundle and measured report are ready.");
 }catch(exc){error(exc.message);toast("Run failed. Completed artifacts are retained.")}
 finally{busy=false;buttons.forEach(b=>{b.disabled=false;b.innerHTML=icon("play")+"Run demo pipeline"})}
}
document.addEventListener("click",event=>{
 const run=event.target.closest(".run-button");if(run){runDemo();return}
 const cond=event.target.closest("[data-condition]");if(cond){condition=cond.dataset.condition;renderCases();return}
 const caseButton=event.target.closest("[data-case-id]");if(caseButton){selectedId=caseButton.dataset.caseId;renderCaseList();return}
});
document.addEventListener("change",event=>{if(event.target.matches("[data-case-choice]")){overviewId=event.target.value;renderCases()}});
["case-search","category-filter","verdict-filter"].forEach(id=>$(id).addEventListener(id==="case-search"?"input":"change",renderCaseList));
$("verify-button").addEventListener("click",async()=>{
 const button=$("verify-button");button.disabled=true;
 try{const result=await api("/api/verify",true);$("verify-result").hidden=false;$("verify-result").textContent="Integrity verified: "+result.verified_artifacts+" artifacts match their recorded SHA-256 values. Run "+result.run_id+"."}
 catch(exc){$("verify-result").hidden=false;$("verify-result").textContent="Integrity check failed: "+exc.message}
 finally{button.disabled=false}
});
window.addEventListener("hashchange",route);icons();route();loadAyaEvidence();load().catch(exc=>error("Could not load the evidence console: "+exc.message));
