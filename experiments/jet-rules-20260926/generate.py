"""Original executable supervision; no benchmark gold labels or cases are sampled."""
import calendar,datetime as dt,hashlib,itertools,json,random
from pathlib import Path
SEED=260926
TREES={
'and_or':{
 'train':[('and',('or',0,1),2),('and',('or',0,1),('or',2,3)),('and',('and',('or',0,1),2),3)],
 'selection':[('and',('or',('not',0),1),2),('and',('or',0,('and',1,2)),3)],
 'test':[('and',('or',0,1),('not',2)),('and',('or',('and',0,1),2),3)]},
'or_not':{
 'train':[('or',('and',0,1),('not',2)),('or',('and',0,('not',1)),('not',2)),('or',('and',0,1),('not',('or',2,3)))],
 'selection':[('or',('and',('not',0),1),('not',2)),('or',('and',0,1),('not',('and',2,3)))],
 'test':[('or',('and',0,('or',1,2)),('not',3)),('or',('and',('or',0,1),2),('not',3))]},
'conditional':{
 'train':[('if',0,('not',1),2),('if',('and',0,1),2,('not',3)),('if',0,('or',1,2),3)],
 'selection':[('if',('not',0),1,2),('if',('or',0,1),('not',2),3)],
 'test':[('if',0,('and',1,2),('not',3)),('if',('and',0,1),('not',2),3)]}}
def evaluate(tree,bits):
 if isinstance(tree,int):return bits[tree]
 op,*args=tree
 if op=='not':return not evaluate(args[0],bits)
 if op=='and':return evaluate(args[0],bits) and evaluate(args[1],bits)
 if op=='or':return evaluate(args[0],bits) or evaluate(args[1],bits)
 if op=='if':return evaluate(args[1],bits) if evaluate(args[0],bits) else evaluate(args[2],bits)
 raise ValueError(op)
def expression(tree):
 if isinstance(tree,int):return f'b[{tree}]'
 op,*args=tree
 if op=='not':return f'(not {expression(args[0])})'
 if op in ['and','or']:return f'({expression(args[0])} {op} {expression(args[1])})'
 return f'({expression(args[1])} if {expression(args[0])} else {expression(args[2])})'
def leaves(tree):
 if isinstance(tree,int):return {tree}
 return set().union(*(leaves(t) for t in tree[1:]))
def render(tree,atoms,style):
 if isinstance(tree,int):return atoms[tree]
 op,*args=tree;a=[render(t,atoms,style) for t in args]
 if op=='not':return ['it is false that ('+a[0]+')','NOT ('+a[0]+')','the following is not satisfied: ('+a[0]+')'][style%3]
 if op=='and':return [f'({a[0]}) AND ({a[1]})',f'both ({a[0]}) and ({a[1]})',f'each condition holds: [{a[0]}]; [{a[1]}]'][style%3]
 if op=='or':return [f'({a[0]}) OR ({a[1]})',f'at least one of ({a[0]}) and ({a[1]})',f'one or both conditions hold: [{a[0]}]; [{a[1]}]'][style%3]
 return [f'if ({a[0]}), require ({a[1]}); otherwise require ({a[2]})',f'when ({a[0]}) holds use ({a[1]}), and when it does not hold use ({a[2]})',f'choose the condition ({a[1]}) if ({a[0]}), or the condition ({a[2]}) otherwise'][style%3]
def atom(rng,name):
 kind=rng.choice(['lt','le','gt','ge','range','flag','match']);k=rng.randint(3,95)
 if kind=='flag':return f'{name} verified is yes',{name+' verified':'yes'},{name+' verified':'no'},kind
 if kind=='match':return f'{name} signer equals {name} approver',{name+' signer':'Ari',name+' approver':'Ari'},{name+' signer':'Ari',name+' approver':'Bo'},kind
 if kind=='range':return f'{name} is between {k} and {k+7}, inclusive',{name:rng.choice([k,k+7,k+3])},{name:rng.choice([k-1,k+8])},kind
 descriptions={'lt':'less than','le':'at most','gt':'greater than','ge':'at least'}
 true={'lt':k-1,'le':k,'gt':k+1,'ge':k};false={'lt':k,'le':k+1,'gt':k,'ge':k-1}
 return f'{name} is {descriptions[kind]} {k}',{name:true[kind]},{name:false[kind]},kind

def decision(state,kind,label,source,group,cert,rng):
 if kind=='score':
  q={'type':'score','instructions':'Determine the applicable lateness category.','criteria':['Arrived on or before the deadline','After deadline but within the allowed grace period','Beyond the grace period; reject']};target=[float(i==label) for i in range(3)]
 elif kind=='noul':q={'type':'noul','instructions':'Does this case meet the stated approval policy?'};target=[float(not label),float(label)]
 else:
  vals=['The case satisfies the approval policy','The case does not satisfy the approval policy'];rng.shuffle(vals)
  q={'type':'choice','instructions':'Decide whether the case meets the policy.','criteria':{f'option_{i}':v for i,v in enumerate(vals)}};target=[float(v==('The case satisfies the approval policy' if label else 'The case does not satisfy the approval policy')) for v in vals]
 return {'state':state,'question':q,'target':target,'source':'rules:'+source,'provenance':{'group':group,'generator_seed':SEED,'certificate':cert}}

def rules(split,family,n):
 rng=random.Random(SEED+sum(map(ord,split+family)));result=[]
 for i in range(n//2):
  tree=TREES[family][split][i%len(TREES[family][split])];nleaves=max(leaves(tree))+1
  values=list(itertools.product([False,True],repeat=nleaves));bits=rng.choice(values);gold=evaluate(tree,bits)
  other=[b for b in values if evaluate(tree,b)!=gold];distance=min(sum(x!=y for x,y in zip(bits,b)) for b in other);opposite=rng.choice([b for b in other if sum(x!=y for x,y in zip(bits,b))==distance])
  names=rng.sample(['invoice amount','ticket count','parcel weight','account balance','review score','shipment units','license level','claim total'],nleaves);spec=[atom(rng,name) for name in names]
  style=rng.choice([0,1]) if split=='train' else 2
  policy='Approval requires '+render(tree,[a[0] for a in spec],style)+'. If the required condition fails, do not approve. The archive reference is irrelevant.'
  for j,b in enumerate([bits,opposite]):
   facts={'archive reference':rng.randint(10000,99999)}
   for v,a in zip(b,spec):facts.update(a[1] if v else a[2])
   items=list(facts.items());rng.shuffle(items);state={'policy':policy,'case':dict(items)}
   assert evaluate(tree,b)==eval(expression(tree),{'__builtins__':{}},{'b':b})
   result.append(decision(state,'noul' if i%5==0 else 'choice',evaluate(tree,b),family,f'{split}:{family}:{i}',{'tree':tree,'truth_values':b,'atom_kinds':[a[3] for a in spec],'label':evaluate(tree,b),'pair_member':j},rng))
 return result

def deadline_label(deadline,received,grace):
 return 0 if received<=deadline else 1 if received<=deadline+dt.timedelta(days=grace) else 2

def deadlines(split,n):
 rng=random.Random(SEED+sum(map(ord,split)));years={'train':list(range(2018,2032)),'selection':list(range(2033,2038)),'test':list(range(2038,2043))}[split];rows=[]
 for i in range(n):
  year=rng.choice(years);month=rng.choice([2,12,1,4,6,9,11] if i%2==0 else list(range(1,13)));day=rng.choice([1,calendar.monthrange(year,month)[1],rng.randint(1,28)]);deadline=dt.date(year,month,day);grace=rng.choice([1,2,3,7,10,14,21,30]);label=i%3
  offset=rng.choice([-7,-1,0]) if label==0 else rng.choice([1,grace]) if label==1 else rng.choice([grace+1,grace+2,grace+14]);received=deadline+dt.timedelta(days=offset)
  fmt=rng.choice(['%Y-%m-%d','%B %d, %Y']) if split=='train' else '%d %B %Y'
  policy=(f'Accept arrivals on or before the due date as on time. A receipt after that date remains acceptable for {grace} calendar days, including the last day. Anything arriving after that grace window must be rejected.' if split=='train' else f'There are three outcomes: on time if receipt is no later than the due date; late but accepted if received after the due date and no more than {grace} calendar days later; rejected otherwise.')
  state={'policy':policy,'case':{'due date':deadline.strftime(fmt),'receipt date':received.strftime(fmt),'tracking reference':rng.randint(100000,999999)}}
  assert deadline_label(deadline,received,grace)==label
  assert label== (0 if offset<=0 else 1 if offset<=grace else 2)
  cert={'deadline':deadline.isoformat(),'received':received.isoformat(),'grace_days':grace,'days_after_deadline':offset,'label':label}
  rows.append(decision(state,'score',label,'deadline',f'{split}:deadline:{i}',cert,rng))
 return rows
