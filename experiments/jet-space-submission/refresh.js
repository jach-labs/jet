const fs=require('fs'),{execFileSync}=require('child_process');
const P='news.html'; let src=fs.readFileSync(P,'utf8');
const js=src.slice(src.indexOf('<script>')+8,src.indexOf('</script>'));
const head=js.slice(0,js.indexOf('/* ---- state ---- */'));
const ITEMS=eval(head+';ITEMS');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const clean=t=>t.split('').filter(c=>{const k=c.charCodeAt(0);return k>=32||c==='\n'||c==='\t'||c==='\r';}).join('');
async function j(url,opt){const r=await fetch(url,opt);if(!r.ok)throw new Error(r.status+' '+url);return JSON.parse(clean(await r.text()));}
/* GitHub stars go through the gh CLI so the whole run fits in one authenticated
   hour (5,000 requests); the unauthenticated API allows only 60. */
const ghStars=repo=>{try{return parseInt(execFileSync('gh',['api','repos/'+repo,'--jq','.stargazers_count'],{encoding:'utf8',stdio:['ignore','pipe','ignore']}).trim(),10);}catch(e){return null;}};
const esc=t=>t.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
const changes=[];
(async()=>{
 for(const it of ITEMS){
  const m={...(it.m||{})}; const L=it.links||{}; const notes=[];
  if(L.tweet){const id=L.tweet.match(/status\/(\d+)/)[1];
    try{const d=await j('https://api.fxtwitter.com/status/'+id);const t=d.tweet;m.likes=t.likes;m.rts=t.retweets;m.views=t.views;}catch(e){notes.push('tweet:'+e.message);}}
  const ghs=[L.gh,L.gh2].filter(Boolean).map(u=>u.match(/github\.com\/([^\/]+\/[^\/#?]+)/)?.[1]).filter(Boolean);
  if(ghs.length){let sum=0,ok=true;for(const r of ghs){const s=ghStars(r);if(s===null){ok=false;notes.push('gh:'+r);}else sum+=s;}if(ok)m.stars=sum;}
  const hfs=[L.hf,L.hf2].filter(Boolean).filter(u=>!u.includes('/collections/')&&!u.includes('/spaces/')).map(u=>u.match(/huggingface\.co\/([^\/]+\/[^\/#?]+)/)?.[1]).filter(Boolean);
  if(hfs.length){let sum=0,ok=true;for(const r of hfs){try{const d=await j('https://huggingface.co/api/models/'+r);sum+=d.likes||0;}catch(e){ok=false;notes.push('hf:'+r);}}if(ok)m.hfLikes=sum;}
  const before=JSON.stringify(it.m||{}),after=JSON.stringify(m);
  if(before!==after||notes.length)changes.push([it.title.slice(0,44).padEnd(44),before,'->',after,notes.join(' ')]);
  const ti=src.indexOf("title: '"+esc(it.title)+"'"); if(ti<0){console.log('TITLE NOT FOUND',it.title);continue;}
  const mi=src.indexOf('m: {',ti); const me=src.indexOf('}',mi)+1;
  const lit='m: { '+Object.entries(m).map(([k,v])=>k+': '+v).join(', ')+' }';
  src=src.slice(0,mi)+lit+src.slice(me);
  await sleep(60);
 }
 fs.writeFileSync(P,src);
 for(const c of changes)console.log(c.join('  '));
 console.log('changed',changes.length,'of',ITEMS.length);
})();
