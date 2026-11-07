"""Two-pass log sum-product on supplied acyclic Ising models with hard evidence."""
import math
from .core import _compile_model,canonical
SPINS=(-1,1)


def _lse(values):
    peak=max(values)
    return peak+math.log(math.fsum(math.exp(x-peak) for x in values)) if peak!=-math.inf else peak


def _evidence(values,n):
    if values is None:return {}
    if not isinstance(values,(list,tuple)) or len(values)>n:raise ValueError('evidence must be a bounded list')
    found={}
    for v in values:
        if not isinstance(v,dict) or set(v)!={'i','spin'}:raise ValueError('evidence requires exactly i and spin')
        i,z=v['i'],v['spin']
        if type(i) is not int or not 0<=i<n or type(z) is not int or z not in SPINS:raise ValueError('evidence requires valid integer index and spin -1 or +1')
        if i in found:raise ValueError('duplicate evidence index')
        found[i]=z
    return found


def _pass(h,edges,adj,parent,order,roots,evidence):
    n=len(h);messages={};trace=[]
    def cavity(i,excluded):
        return [(-math.inf if i in evidence and evidence[i]!=z else
                 math.fsum([h[i]*z]+[messages[(k,i)][s] for k,w in adj[i] if k!=excluded])) for s,z in enumerate(SPINS)]
    def send(i,j,w):
        c=cavity(i,j);out=[_lse([c[s]+w*z*y for s,z in enumerate(SPINS)]) for y in SPINS]
        messages[(i,j)]=out
        trace.append({'from':i,'to':j,'log_values':out[:],
                      'depends_on':sorted(k for k,v in adj[i] if k!=j)})
    for i in reversed(order):
        if parent[i] is not None:
            j,w=parent[i];send(i,j,w)
    for i in order:
        for j,w in adj[i]:
            if parent[j] is not None and parent[j][0]==i:send(i,j,w)
    log_components=[{'root':i,'log_partition':_lse(cavity(i,None))} for i in roots]
    singleton=[]
    for i in range(n):
        c=cavity(i,None);norm=_lse(c);logs=[v-norm if v!=-math.inf else None for v in c]
        p=[math.exp(v) if v is not None else 0.0 for v in logs]
        singleton.append({'i':i,'probabilities':p,'log_probabilities':logs,'mean':p[1]-p[0]})
    joints=[]
    for i,j,w in edges:
        ci,cj=cavity(i,j),cavity(j,i);states=[(x,y) for x in SPINS for y in SPINS]
        values=[ci[SPINS.index(x)]+cj[SPINS.index(y)]+w*x*y for x,y in states];norm=_lse(values)
        logs=[v-norm if v!=-math.inf else None for v in values];p=[math.exp(v) if v is not None else 0.0 for v in logs]
        moment=math.fsum(a*x*y for a,(x,y) in zip(p,states))
        joints.append({'i':i,'j':j,'states':[{'state':list(z),'probability':a,'log_probability':b} for z,a,b in zip(states,p,logs)],
                       'second_moment':moment,'covariance':moment-singleton[i]['mean']*singleton[j]['mean']})
    return {'log_partition':math.fsum(c['log_partition'] for c in log_components),'components':log_components,
            'singletons':singleton,'edge_joints':joints,'directed_messages':trace,'message_count':len(trace)}


def forest(payload,*,evidence=None):
    """Exact tree sum-product in floating-point log space, not loopy BP.

    The forest path allows 1..128 nodes; finite enumeration and mean field keep
    their existing independent 12-node limits. Cycles (even zero edges) fail.
    """
    h,edges=_compile_model(payload,128);n=len(h)
    if len(edges)>n-1:raise ValueError('forest requires an acyclic graph')
    given=_evidence(evidence,n);adj=[[] for _ in h];uf=list(range(n))
    def find(i):
        while uf[i]!=i:uf[i]=uf[uf[i]];i=uf[i]
        return i
    for i,j,w in edges:
        a,b=find(i),find(j)
        if a==b:raise ValueError('forest requires an acyclic graph')
        uf[a]=b;adj[i].append((j,w));adj[j].append((i,w))
    parent=[None]*n;seen=set();order=[];roots=[]
    for root in range(n):
        if root in seen:continue
        roots.append(root);seen.add(root);stack=[root]
        while stack:
            i=stack.pop();order.append(i)
            for j,w in reversed(adj[i]):
                if j not in seen:seen.add(j);parent[j]=(i,w);stack.append(j)
    conditional=_pass(h,edges,adj,parent,order,roots,given)
    unconditional=_pass(h,edges,adj,parent,order,roots,{}) if given else conditional
    log_evidence=conditional['log_partition']-unconditional['log_partition'] if given else 0.0
    return {'format':'varilab-forest-v1','model':canonical(h,edges),
            'evidence':[{'i':i,'spin':z} for i,z in sorted(given.items())],
            'state_order':[-1,1],'roots':roots,'traversal_order':order,
            'conditional':conditional,'unconditional_log_partition':unconditional['log_partition'],
            'log_evidence_probability':log_evidence,'evidence_probability':math.exp(log_evidence),
            'scope':'acyclic supplied Ising graph; exact sum-product structure using floating-point log-sum-exp, not rational arithmetic; conditional singleton and edge marginals only; message log values are unnormalized subtree sums, not probabilities; impossible clamped states have zero mass and null log probability; underflowed possible probabilities retain finite logs'}


def forest_demo():
    payload={'version':1,'fields':[0,0,0,0],'edges':[{'i':i,'j':i+1,'weight':2} for i in range(3)]}
    prior=forest(payload);conditioned=forest(payload,evidence=[{'i':0,'spin':1}])
    expected=[1]+[math.tanh(2)**i for i in range(1,4)]
    actual=[r['mean'] for r in conditioned['conditional']['singletons']]
    assert max(abs(a-b) for a,b in zip(expected,actual))<1e-12
    assert all(abs(r['mean'])<1e-12 for r in prior['conditional']['singletons'])
    assert abs(conditioned['evidence_probability']-.5)<1e-12
    large={'version':1,'fields':[0]*128,'edges':[{'i':i,'j':i+1,'weight':2} for i in range(127)]}
    long=forest(large,evidence=[{'i':0,'spin':1}])
    assert long['conditional']['message_count']==254
    return {'prior':prior,'conditioned':conditioned,'long_chain':long,
            'analytic_conditional_means':expected,'gates':{'conditional_endpoint_mean_gt_point8':actual[-1]>.8,
            'unconditioned_endpoint_mean_gt_point8':prior['conditional']['singletons'][-1]['mean']>.8},
            'selection':'direct four-node zero-field chain J=2 with first spin +1, plus 128-node chain with identical fields/couplings; no parameter search',
            'scope':'supplied-model conditioning, not a causal intervention or observationally learned dependence; ignoring evidence is an intentionally different query, not an inference-method benchmark'}
