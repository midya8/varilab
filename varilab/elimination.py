"""Bounded log-domain variable elimination with symbolic preflight and min-fill."""
import itertools,math
from .core import _compile_model,canonical
from .forest import _evidence,_lse,SPINS
MAX_BUCKET_STATES=4096
MAX_FACTOR_READS=250000
MAX_OUTPUT_ENTRIES=65536


def _prepare(payload,query,evidence,order):
    h,edges=_compile_model(payload,64);n=len(h);given=_evidence(evidence,n)
    if query is None:query=[]
    if not isinstance(query,(list,tuple)) or len(query)>8 or any(type(i) is not int or not 0<=i<n for i in query) or len(set(query))!=len(query):raise ValueError('query must contain at most 8 distinct integer indices')
    if any(i in given for i in query):raise ValueError('query and evidence must be disjoint')
    query=sorted(query);free=set(range(n))-set(given)-set(query)
    if order is not None and (not isinstance(order,(list,tuple)) or len(order)!=len(free) or any(type(i) is not int for i in order) or set(order)!=free):raise ValueError('order must permute all unobserved nonquery indices')
    scopes={f'u{i}':tuple([i] if i not in given else []) for i in range(n)}
    scopes.update({f'p{k}':tuple(i for i in (a,b) if i not in given) for k,(a,b,w) in enumerate(edges)})
    initial_entries=sum(2**len(s) for s in scopes.values());stored=initial_entries;reads=0;peak=max([2**len(s) for s in scopes.values()]+[1]);steps=[];chosen=[]
    remaining=set(free)
    while remaining:
        if order is None:
            adjacency={i:set() for i in remaining|set(query)}
            for scope in scopes.values():
                for a,b in itertools.combinations(scope,2):adjacency[a].add(b);adjacency[b].add(a)
            def priority(i):
                neighbors=adjacency[i]
                return (sum(b not in adjacency[a] for a,b in itertools.combinations(sorted(neighbors),2)),len(neighbors),i)
            i=min(remaining,key=priority)
        else:i=order[len(chosen)]
        ids=[k for k,s in scopes.items() if i in s];bag=sorted(set().union(*(set(scopes[k]) for k in ids)));out=tuple(j for j in bag if j!=i)
        count=2**len(bag);cost=count*len(ids);name=f'e{len(steps)}';output_count=2**len(out)
        steps.append({'eliminate':i,'input_factor_ids':ids,'bag':bag,'assignment_count':count,'output_factor_id':name,'output_scope':list(out),'output_entries':output_count,'factor_entry_reads':cost})
        reads+=cost;stored+=output_count;peak=max(peak,count);chosen.append(i);remaining.remove(i)
        for k in ids:del scopes[k]
        scopes[name]=out
    final_ids=list(scopes);final_entries=2**len(query);final_reads=final_entries*len(final_ids);reads+=final_reads;stored+=final_entries;peak=max(peak,final_entries)
    reasons=[]
    if peak>MAX_BUCKET_STATES:reasons.append('peak bucket assignments exceed cap')
    if reads>MAX_FACTOR_READS:reasons.append('factor-entry reads exceed cap')
    if stored>MAX_OUTPUT_ENTRIES:reasons.append('initial/intermediate/final output entries exceed cap')
    plan={'format':'varilab-elimination-plan-v1','query':query,'evidence':[{'i':i,'spin':z} for i,z in sorted(given.items())],
          'order_strategy':'min-fill heuristic, degree then index tie break; not optimal treewidth' if order is None else 'explicit order',
          'order':chosen,'steps':steps,'initial_entries':initial_entries,'total_output_entries':stored,
          'factor_entry_reads':reads,'peak_bucket_states':peak,'max_eliminated_neighbors':max([len(s['bag'])-1 for s in steps]+[0]),
          'final_factor_ids':final_ids,'final_scope':query,'final_entries':final_entries,'final_factor_entry_reads':final_reads,
          'limits':{'bucket_states':MAX_BUCKET_STATES,'factor_entry_reads':MAX_FACTOR_READS,'output_entries':MAX_OUTPUT_ENTRIES},
          'accepted':not reasons,'rejection_reasons':reasons,
          'accounting':'symbolic assignment and scalar factor-entry-read counts for this implementation, not FLOPs, elapsed time, memory bytes or optimal treewidth; stored zero-weight edges are not removed; includes initial/intermediate/final factor output entries, excluding transient scalar calculations'}
    return h,edges,given,query,plan


def elimination_plan(payload,*,query=None,evidence=None,order=None):
    """Plan scopes and resource counters without building numeric factor tables."""
    return _prepare(payload,query,evidence,order)[4]


def _initial_factors(h,edges,given):
    factors={}
    for i,a in enumerate(h):
        scope=() if i in given else (i,)
        factors[f'u{i}']=(scope,[a*given[i]] if not scope else [-a,a])
    for k,(a,b,w) in enumerate(edges):
        scope=tuple(i for i in (a,b) if i not in given);values=[]
        for state in itertools.product(SPINS,repeat=len(scope)):
            z=dict(zip(scope,state));z.update(given);values.append(w*z[a]*z[b])
        factors[f'p{k}']=(scope,values)
    return factors


def _index(scope,z):
    value=0
    for i in scope:value=2*value+(z[i]==1)
    return value


def _execute(h,edges,given,query,plan):
    factors=_initial_factors(h,edges,given);trace=[];reads=0;entries=sum(len(v) for s,v in factors.values())
    for step in plan['steps']:
        ids=step['input_factor_ids'];scope=tuple(step['output_scope']);out=[]
        for state in itertools.product(SPINS,repeat=len(scope)):
            z=dict(zip(scope,state));candidates=[]
            for spin in SPINS:
                z[step['eliminate']]=spin
                candidates.append(math.fsum(factors[k][1][_index(factors[k][0],z)] for k in ids));reads+=len(ids)
            out.append(_lse(candidates))
        for k in ids:del factors[k]
        factors[step['output_factor_id']]=(scope,out);entries+=len(out)
        trace.append({'factor_id':step['output_factor_id'],'scope':list(scope),'log_values':out[:]})
    values=[];states=list(itertools.product(SPINS,repeat=len(query)))
    for state in states:
        z=dict(zip(query,state));values.append(math.fsum(factors[k][1][_index(factors[k][0],z)] for k in plan['final_factor_ids']));reads+=len(factors)
    entries+=len(values);logz=_lse(values)
    assert reads==plan['factor_entry_reads'] and entries==plan['total_output_entries']
    return {'log_partition':logz,'query_states':[{'state':list(z),'log_probability':v-logz,'probability':math.exp(v-logz)} for z,v in zip(states,values)],
            'final_log_values':values,'factor_trace':trace,'executed_factor_entry_reads':reads,'executed_output_entries':entries}


def eliminate(payload,*,query=None,evidence=None,order=None):
    """Evidence-restricted partition and optional conditional query joint table."""
    h,edges,given,query,plan=_prepare(payload,query,evidence,order)
    if not plan['accepted']:raise ValueError('elimination rejected before numeric factor tables: '+ '; '.join(plan['rejection_reasons']))
    result=_execute(h,edges,given,query,plan)
    return {'format':'varilab-elimination-v1','model':canonical(h,edges),'plan':plan,**result,
            'scope':'supplied binary Ising graph, including cycles, 1..64 spins and at most 8 unobserved query spins; log_partition sums exp(E) over evidence-compatible states and is not normalized evidence probability; query_states is the normalized conditional joint in sorted query-index order, rightmost spin fastest; float log-sum-exp, not rational arithmetic; no clipping, parameter learning, causal or treewidth-optimality claim'}


def elimination_demo():
    star={'version':1,'fields':[0]*16,'edges':[{'i':0,'j':i,'weight':.5} for i in range(1,16)]}
    bad=elimination_plan(star,order=list(range(16)));good=eliminate(star)
    assert not bad['accepted'] and bad['peak_bucket_states']==65536 and good['plan']['peak_bucket_states']==4
    star_analytic=math.log(2)+15*math.log(2*math.cosh(.5));assert abs(good['log_partition']-star_analytic)<1e-12
    cycle={'version':1,'fields':[0]*64,'edges':[{'i':i,'j':i+1,'weight':.5} for i in range(63)]+[{'i':0,'j':63,'weight':.5}]}
    ring=eliminate(cycle,query=[0,1]);t=math.tanh(.5);ring_analytic=64*math.log(2*math.cosh(.5))+math.log1p(t**64)
    assert abs(ring['log_partition']-ring_analytic)<1e-11 and ring['plan']['peak_bucket_states']==8
    moment=math.fsum(row['probability']*math.prod(row['state']) for row in ring['query_states']);analytic_moment=(t+t**63)/(1+t**64);assert abs(moment-analytic_moment)<1e-12
    return {'hub_first_plan':bad,'min_fill_star':good,'cycle_query':ring,
            'analytic_star_log_partition':star_analytic,'analytic_cycle_log_partition':ring_analytic,
            'cycle_edge_moment':moment,'analytic_cycle_edge_moment':analytic_moment,
            'gates':{'hub_first_plan_accepted':bad['accepted'],'min_fill_star_plan_accepted':good['plan']['accepted'],'cycle_partition_matches_analytic':abs(ring['log_partition']-ring_analytic)<1e-11},
            'selection':'direct zero-field 16-node star and 64-node cycle, uniform coupling 1/2; explicit hub-first order compared with deterministic min-fill heuristic; no order search or runtime measurement',
            'scope':'the same star partition query has different symbolic workload under different orders; a rejected order is not an impossible graph; min-fill is a heuristic, not minimum treewidth; cyclic query checked by a separate analytic zero-field ring formula'}
