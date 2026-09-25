const fs=require('fs');
const h=fs.readFileSync('news.html','utf8');
const js=h.slice(h.indexOf('<script>')+8,h.indexOf('</script>'));
const head=js.slice(0,js.indexOf('/* ---- state ---- */'));
const checks=`
console.log('CATS',CATS.map(c=>c.order+':'+c.id).join(' '),'| ITEMS',ITEMS.length,'| orphans',ITEMS.filter(i=>!CATS.find(c=>c.id===i.cat)).length,'| maxdesc',Math.max(...ITEMS.map(i=>i.desc.length)));
for(const c of CATS){console.log(' ',c.id,'->',sortItems(ITEMS.filter(i=>i.cat===c.id),'trending').slice(0,5).map(i=>i.title.slice(0,26)+'='+trending(i)).join(' | '))}
`;
eval(head+checks);
