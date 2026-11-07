"""Exact binary-rational forest max-sum, multiplicities and compatible traceback."""
from fractions import Fraction as F
from math import prod
from .core import canonical
from .forest import _prepare,SPINS


def _pack(x):
    return None if x is None else {'numerator':x.numerator,'denominator':x.denominator}


def forest_map(payload,*,evidence=None):
    """MAP scores and counts for compiled binary floats, with no tie tolerance.

    Scores are unnormalized log potentials, not probabilities. For forbidden
    states None represents negative infinity; its maximizing count is zero.
    A deterministic first-spin tie break selects one compatible optimum, not
    necessarily the lexicographically first configuration in index order.
    """
    h,edges,given,adj,parent,order,roots=_prepare(payload,evidence)
    original=canonical(h,edges);h=list(map(F,h));edges=[(i,j,F(w)) for i,j,w in edges]
    adj=[[(j,F(w)) for j,w in row] for row in adj]
    messages={};choices={};trace=[]
    def cavity(i,excluded):
        rows=[]
        for s,z in enumerate(SPINS):
            if i in given and given[i]!=z:rows.append((None,0));continue
            incoming=[messages[(k,i)][s] for k,w in adj[i] if k!=excluded]
            rows.append((h[i]*z+sum(v for v,c in incoming),prod(c for v,c in incoming)))
        return rows
    def best(candidates):
        possible=[(z,v,c) for z,v,c in candidates if v is not None]
        score=max(v for z,v,c in possible)
        tied=[(z,c) for z,v,c in possible if v==score]
        return score,sum(c for z,c in tied),[z for z,c in tied]
    def send(i,j,w):
        c=cavity(i,j);out=[];back=[]
        for y in SPINS:
            score,count,tied=best([(z,v+w*z*y if v is not None else None,count) for z,(v,count) in zip(SPINS,c)])
            out.append((score,count));back.append(tied)
        messages[(i,j)]=out;choices[(i,j)]=back
        trace.append({'from':i,'to':j,'scores':[_pack(v) for v,c in out],
                      'maximizing_counts':[c for v,c in out], 'maximizing_source_spins':back,
                      'depends_on':sorted(k for k,w in adj[i] if k!=j)})
    for i in reversed(order):
        if parent[i] is not None:
            j,w=parent[i];send(i,j,F(w))
    for i in order:
        for j,w in adj[i]:
            if parent[j] is not None and parent[j][0]==i:send(i,j,w)
    components=[];component_of={};root_best={};assignment=[None]*len(h);backtrack=[]
    for i in order:
        root=i if parent[i] is None else component_of[parent[i][0]];component_of[i]=root
        if i==root:
            score,count,tied=best([(z,v,c) for z,(v,c) in zip(SPINS,cavity(i,None))])
            root_best[root]=(score,count);assignment[i]=tied[0]
            components.append({'root':root,'score':_pack(score),'maximizing_count':count,'maximizing_root_spins':tied})
            backtrack.append({'i':i,'parent':None,'parent_spin':None,'chosen_spin':tied[0],'compatible_spins':tied[:]})
        else:
            j,w=parent[i];tied=choices[(i,j)][SPINS.index(assignment[j])];assignment[i]=tied[0]
            backtrack.append({'i':i,'parent':j,'parent_spin':assignment[j],'chosen_spin':tied[0],'compatible_spins':tied[:]})
    optimum=sum(v for v,c in root_best.values());multiplicity=prod(c for v,c in root_best.values());max_marginals=[]
    for i in range(len(h)):
        root=component_of[i];other_score=optimum-root_best[root][0];other_count=multiplicity//root_best[root][1]
        rows=[]
        for z,(v,c) in zip(SPINS,cavity(i,None)):
            total=v+other_score if v is not None else None
            rows.append({'spin':z,'score':_pack(total),'maximizing_count':c*other_count,
                         'gap_to_optimum':_pack(optimum-total) if total is not None else None,
                         'attains_global_optimum':total==optimum})
        max_marginals.append({'i':i,'states':rows})
    witness=sum(a*z for a,z in zip(h,assignment))+sum(w*assignment[i]*assignment[j] for i,j,w in edges)
    assert witness==optimum and all(assignment[i]==z for i,z in given.items())
    return {'format':'varilab-forest-map-v1','model':original,
            'evidence':[{'i':i,'spin':z} for i,z in sorted(given.items())],
            'state_order':list(SPINS),'roots':roots,'traversal_order':order,
            'assignment':assignment,'optimal_score':_pack(optimum),'assignment_score':_pack(witness),
            'maximizing_count':multiplicity,'unique':multiplicity==1,'components':components,
            'max_marginals':max_marginals,'directed_messages':trace,'message_count':len(trace),'backtrack':backtrack,
            'tie_rule':'first spin in [-1,+1] at each root, then first compatible stored source spin conditional on its chosen parent; deterministic for canonical graph traversal, not globally lexicographic',
            'scope':'acyclic supplied Ising graph with hard evidence, 1..128 spins; exact rational max-sum and tie counts for the compiled binary-float potentials, not original decimal strings, probabilities, learned parameters, causal claims or loopy approximation'}


def map_demo():
    from .forest import forest
    p={'version':1,'fields':[.25,.24609375],'edges':[{'i':0,'j':1,'weight':-2}]}
    posterior=forest(p);decoded=forest_map(p)
    modes=[-1 if row['probabilities'][0]>=row['probabilities'][1] else 1 for row in posterior['conditional']['singletons']]
    marginal_score=sum(F(a)*z for a,z in zip(p['fields'],modes))+F(-2)*modes[0]*modes[1]
    assert modes==[1,1] and decoded['assignment']==[1,-1] and decoded['unique']
    assert decoded['optimal_score']==_pack(F(513,256)) and marginal_score==F(-385,256)
    tied=forest_map({'version':1,'fields':[0,0],'edges':[{'i':0,'j':1,'weight':-2}]})
    assert tied['maximizing_count']==2 and tied['assignment']==[-1,1]
    naive=[row['states'][0]['spin'] for row in tied['max_marginals']]
    assert naive==[-1,-1]
    large=forest_map({'version':1,'fields':[0]*128,'edges':[]})
    assert large['maximizing_count']==2**128
    return {'unique_case':decoded,'marginal_probabilities':[r['probabilities'] for r in posterior['conditional']['singletons']],
            'marginal_modes':modes,'marginal_mode_score':_pack(marginal_score),
            'score_gap':_pack(F(513,256)-marginal_score),'tied_case':tied,'naive_local_max_mode_assignment':naive,
            'independent_128':{'maximizing_count':large['maximizing_count'],'assignment':large['assignment']},
            'gates':{'decoded_is_optimal':decoded['assignment_score']==decoded['optimal_score'],
                     'marginal_modes_are_optimal':marginal_score==F(513,256)},
            'selection':'directly constructed near-equal dyadic positive fields 1/4 and 63/256, antiferromagnetic coupling -2; separate symmetric zero-field tie and 128 isolated zero-field spins',
            'scope':'marginal modes and joint MAP answer different decision questions; this constructed example is not a method ranking; exact rational score certificates do not certify floating marginal probabilities'}
