"""Bounded exact-rational max-sum elimination, multiplicities and traceback."""
import itertools
from fractions import Fraction as F
from math import prod
from .core import canonical
from .forest import SPINS
from .elimination import _prepare,_index
from .decoding import _pack


def _initial(h,edges,given):
    factors={}
    for i,a in enumerate(h):
        scope=() if i in given else (i,)
        factors[f'u{i}']=(scope,[(a*given[i],1)] if not scope else [(-a,1),(a,1)])
    for k,(a,b,w) in enumerate(edges):
        scope=tuple(i for i in (a,b) if i not in given);values=[]
        for state in itertools.product(SPINS,repeat=len(scope)):
            z=dict(zip(scope,state));z.update(given);values.append((w*z[a]*z[b],1))
        factors[f'p{k}']=(scope,values)
    return factors


def cyclic_map(payload,*,query=None,evidence=None,order=None):
    """Maximize all unobserved spins; query indexes expose constrained optima.

    This is joint MAP, NOT marginal MAP: queries also maximize at the last step.
    Every tie is an exact rational equality of compiled binary-float potentials.
    """
    h,edges,given,query,plan=_prepare(payload,query,evidence,order)
    if not plan['accepted']:raise ValueError('cyclic MAP rejected before numeric factor tables: '+ '; '.join(plan['rejection_reasons']))
    original=canonical(h,edges);h=list(map(F,h));edges=[(i,j,F(w)) for i,j,w in edges]
    factors=_initial(h,edges,given);trace=[];backs=[];reads=0;entries=sum(len(v) for s,v in factors.values())
    for step in plan['steps']:
        ids=step['input_factor_ids'];scope=tuple(step['output_scope']);out=[];choices=[]
        for state in itertools.product(SPINS,repeat=len(scope)):
            z=dict(zip(scope,state));candidates=[]
            for spin in SPINS:
                z[step['eliminate']]=spin
                rows=[factors[k][1][_index(factors[k][0],z)] for k in ids];reads+=len(ids)
                candidates.append((spin,sum(v for v,c in rows),prod(c for v,c in rows)))
            score=max(v for s,v,c in candidates);tied=[s for s,v,c in candidates if v==score]
            out.append((score,sum(c for s,v,c in candidates if v==score)));choices.append(tied)
        for k in ids:del factors[k]
        factors[step['output_factor_id']]=(scope,out);entries+=len(out);backs.append((step['eliminate'],scope,choices))
        trace.append({'factor_id':step['output_factor_id'],'eliminate':step['eliminate'],'scope':list(scope),
                      'scores':[_pack(v) for v,c in out],'maximizing_counts':[c for v,c in out],
                      'maximizing_eliminated_spins':choices})
    states=list(itertools.product(SPINS,repeat=len(query)));values=[]
    for state in states:
        z=dict(zip(query,state));rows=[factors[k][1][_index(factors[k][0],z)] for k in plan['final_factor_ids']];reads+=len(rows)
        values.append((sum(v for v,c in rows),prod(c for v,c in rows)))
    entries+=len(values);optimum=max(v for v,c in values);count=sum(c for v,c in values if v==optimum)
    chosen=next(s for s,(v,c) in zip(states,values) if v==optimum);z=dict(given);z.update(zip(query,chosen));backtrack=[]
    for i,scope,choices in reversed(backs):
        context=[z[j] for j in scope];tied=choices[_index(scope,z)];z[i]=tied[0]
        backtrack.append({'i':i,'context_scope':list(scope),'context_state':context,'compatible_spins':tied[:],'chosen_spin':z[i]})
    assignment=[z[i] for i in range(len(h))]
    witness=sum(a*z for a,z in zip(h,assignment))+sum(w*assignment[i]*assignment[j] for i,j,w in edges)
    assert witness==optimum and all(assignment[i]==s for i,s in given.items())
    assert reads==plan['factor_entry_reads'] and entries==plan['total_output_entries']
    return {'format':'varilab-cyclic-map-v1','model':original,'plan':plan,'assignment':assignment,
            'optimal_score':_pack(optimum),'assignment_score':_pack(witness),'maximizing_count':count,'unique':count==1,
            'query_states':[{'state':list(s),'score':_pack(v),'maximizing_count':c,'gap_to_optimum':_pack(optimum-v),'attains_global_optimum':v==optimum} for s,(v,c) in zip(states,values)],
            'chosen_query_state':list(chosen),'factor_trace':trace,'backtrack':backtrack,
            'executed_factor_entry_reads':reads,'executed_output_entries':entries,
            'tie_rule':'first maximizing query state in sorted-index [-1,+1] product order, then first compatible stored spin in reverse elimination order; deterministic for this plan, not a globally lexicographic or order-invariant witness',
            'scope':'supplied binary Ising graph including cycles, 1..64 spins and at most 8 unobserved query spins; joint MAP with constrained max scores, NOT marginal MAP (no summation) or probabilities; exact fractions of compiled binary-float potentials, not original decimal strings; integer tie counts; shared symbolic table-work caps exclude variable-bit rational/count arithmetic and serialized bytes; not minimum treewidth, learning or causal inference'}


def cyclic_map_demo():
    # An odd antiferromagnetic ring must leave at least one edge unsatisfied.
    triangle={'version':1,'fields':[0]*3,'edges':[{'i':0,'j':1,'weight':-1},{'i':0,'j':2,'weight':-1},{'i':1,'j':2,'weight':-1}]}
    small=cyclic_map(triangle,query=[0,1]);singletons=[cyclic_map(triangle,query=[i])['query_states'] for i in range(3)]
    naive=[next(row['state'][0] for row in rows if row['attains_global_optimum']) for rows in singletons]
    naive_score=sum(F(e['weight'])*naive[e['i']]*naive[e['j']] for e in triangle['edges'])
    assert small['optimal_score']==_pack(F(1)) and small['maximizing_count']==6
    ring={'version':1,'fields':[0]*63,'edges':[{'i':i,'j':i+1,'weight':-1} for i in range(62)]+[{'i':0,'j':62,'weight':-1}]}
    large=cyclic_map(ring,query=[0,1]);assert large['optimal_score']==_pack(F(61)) and large['maximizing_count']==126 and large['plan']['peak_bucket_states']==8
    from .elimination import elimination_plan
    star={'version':1,'fields':[0]*16,'edges':[{'i':0,'j':i,'weight':.5} for i in range(1,16)]}
    bad=elimination_plan(star,order=list(range(16)));good=cyclic_map(star)
    assert not bad['accepted'] and good['maximizing_count']==2 and good['optimal_score']==_pack(F(15,2))
    return {'triangle':small,'singleton_max_score_certificates':singletons,'naive_local_first_spins':naive,'naive_score':_pack(naive_score),'score_gap':_pack(F(4)),
            'odd_63_cycle':large,'hub_first_plan':bad,'min_fill_star':good,
            'gates':{'compatible_traceback_attains_optimum':small['assignment_score']==small['optimal_score'],'independent_local_first_spins_attain_optimum':naive_score==F(1),'hub_first_plan_accepted':bad['accepted'],'min_fill_plan_accepted':good['plan']['accepted']},
            'selection':'direct zero-field all-weight-minus-one triangle and odd 63-cycle, and zero-field 16-star with coupling 1/2; no search over parameters or elimination orders',
            'analytic_reference':'an odd antiferromagnetic n-cycle of unit magnitude has score n-2 and 2*n optima: choose its one unsatisfied edge and a global spin flip; triangle gives 1 and 6, ring gives 61 and 126; a positive 16-star has score 15/2 and two aligned optima',
            'scope':'compatible traceback versus independently picking the first locally optimal spin, not a probabilistic marginal comparison, wall-time benchmark or general method ranking'}
