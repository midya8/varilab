import importlib
import copy,itertools,json,math,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from varilab import cyclic_map,forest_map
from varilab.cyclic_map import cyclic_map_demo
from varilab.elimination import elimination_plan
from varilab.cyclic_map import _initial


def fraction(d):return F(d['numerator'],d['denominator'])
def model(h,ws):
    return {'version':1,'fields':list(h),'edges':[{'i':i,'j':j,'weight':w} for (i,j),w in zip(itertools.combinations(range(len(h)),2),ws)]}
def energy(p,z):return sum(F(float(a))*s for a,s in zip(p['fields'],z))+sum(F(float(e['weight']))*z[e['i']]*z[e['j']] for e in p['edges'])
def oracle(p,evidence,query):
    given={r['i']:r['spin'] for r in evidence};buckets={}
    for z in itertools.product([-1,1],repeat=len(p['fields'])):
        if any(z[i]!=s for i,s in given.items()):continue
        key=tuple(z[i] for i in sorted(query));v=energy(p,z)
        old=buckets.get(key)
        if old is None or v>old[0]:buckets[key]=(v,1)
        elif v==old[0]:buckets[key]=(v,old[1]+1)
    best=max(v for v,c in buckets.values());return buckets,best,sum(c for v,c in buckets.values() if v==best)


class CyclicMapTests(unittest.TestCase):
    def assert_reference(self,p,evidence=(),query=(),order=None):
        r=cyclic_map(p,evidence=list(evidence),query=list(query),order=order);rows,best,count=oracle(p,evidence,query)
        self.assertEqual(fraction(r['optimal_score']),best);self.assertEqual(r['maximizing_count'],count);self.assertEqual(r['unique'],count==1)
        self.assertEqual(fraction(r['assignment_score']),energy(p,r['assignment']));self.assertEqual(energy(p,r['assignment']),best)
        self.assertTrue(all(r['assignment'][x['i']]==x['spin'] for x in evidence))
        for row in r['query_states']:
            v,c=rows[tuple(row['state'])];self.assertEqual(fraction(row['score']),v);self.assertEqual(row['maximizing_count'],c)
            self.assertEqual(fraction(row['gap_to_optimum']),best-v);self.assertEqual(row['attains_global_optimum'],v==best)
        self.assertEqual(r['executed_factor_entry_reads'],r['plan']['factor_entry_reads']);self.assertEqual(r['executed_output_entries'],r['plan']['total_output_entries'])
        return r
    def test_972_cyclic_exhaustive_combinations(self):
        checks=0
        for h in itertools.product([-.25,0,.25],repeat=3):
            for ws in [(-1,-1,-1),(.5,-1,0),(0,0,0)]:
                p=model(h,ws)
                for ev in [[],[{'i':0,'spin':-1}],[{'i':0,'spin':1}]]:
                    free=[i for i in range(3) if i not in [r['i'] for r in ev]]
                    for q in [[],[free[0]]]:
                        for order in [None,list(reversed([i for i in free if i not in q]))]:
                            self.assert_reference(p,ev,q,order);checks+=1
        self.assertEqual(checks,972)
    def test_all_24_dense_orders(self):
        p=model([.25,0,-.5,.125],[.5,-1,.25,0,-.25,1])
        for order in itertools.permutations(range(4)):self.assert_reference(p,order=list(order))
    def test_each_intermediate_factor_semantics(self):
        p=model([.25,-.5,.125,0],[.5,-1,.25,-.5,.125,1]);ev=[{'i':3,'spin':1}]
        r=self.assert_reference(p,ev,[2],order=[1,0]);given={3:1}
        terms={f'u{i}':([i],lambda z,i=i:F(float(p['fields'][i]))*z[i]) for i in range(4)}
        terms.update({f'p{k}':([e['i'],e['j']],lambda z,e=e:F(float(e['weight']))*z[e['i']]*z[e['j']]) for k,e in enumerate(p['edges'])})
        ancestry={k:({k},set()) for k in terms}
        for step,trace in zip(r['plan']['steps'],r['factor_trace']):
            ids=step['input_factor_ids'];source=set().union(*(ancestry[k][0] for k in ids));internal=set().union(*(ancestry[k][1] for k in ids))|{step['eliminate']}
            scope=trace['scope']
            for idx,state in enumerate(itertools.product([-1,1],repeat=len(scope))):
                scores=[]
                for hidden in itertools.product([-1,1],repeat=len(internal)):
                    z=dict(given);z.update(zip(scope,state));z.update(zip(sorted(internal),hidden));scores.append(sum(terms[k][1](z) for k in source))
                best=max(scores);self.assertEqual(fraction(trace['scores'][idx]),best);self.assertEqual(trace['maximizing_counts'][idx],scores.count(best))
            for k in ids:del ancestry[k]
            ancestry[step['output_factor_id']]=(source,internal)
    def test_traceback_context_and_compatibility(self):
        r=cyclic_map_demo()['triangle'];trace={x['eliminate']:x for x in r['factor_trace']}
        for row in r['backtrack']:
            self.assertEqual(row['context_state'],[r['assignment'][i] for i in row['context_scope']]);self.assertEqual(row['chosen_spin'],row['compatible_spins'][0])
            idx=0
            for s in row['context_state']:idx=2*idx+(s==1)
            self.assertEqual(row['compatible_spins'],trace[row['i']]['maximizing_eliminated_spins'][idx])
    def test_demo_analytic_and_both_gates(self):
        r=cyclic_map_demo();self.assertEqual(r['triangle']['maximizing_count'],6);self.assertEqual(r['odd_63_cycle']['maximizing_count'],126)
        self.assertEqual(r['gates'],{'compatible_traceback_attains_optimum':True,'independent_local_first_spins_attain_optimum':False,'hub_first_plan_accepted':False,'min_fill_plan_accepted':True})
    def test_odd_and_even_rings_analytic(self):
        for n in range(3,13):
            p={'version':1,'fields':[0]*n,'edges':[{'i':i,'j':i+1,'weight':-1} for i in range(n-1)]+[{'i':0,'j':n-1,'weight':-1}]}
            r=cyclic_map(p,query=[0,1]);self.assertEqual(fraction(r['optimal_score']),n-2 if n%2 else n);self.assertEqual(r['maximizing_count'],2*n if n%2 else 2)
    def test_all_observed(self):self.assert_reference(model([.25,-.5,0],[-1,.5,1]),[{'i':i,'spin':1} for i in range(3)])
    def test_all_query(self):self.assert_reference(model([.25,-.5,0],[-1,.5,1]),query=[2,0,1])
    def test_isolated_64_count(self):
        r=cyclic_map({'version':1,'fields':[0]*64,'edges':[]},query=[3,1]);self.assertEqual(r['maximizing_count'],2**64)
        self.assertTrue(all(x['maximizing_count']==2**62 for x in r['query_states']));self.assertEqual(r['assignment'],[-1]*64)
    def test_disconnected_count_product(self):
        p={'version':1,'fields':[0]*7,'edges':[{'i':a,'j':b,'weight':-1} for a,b in [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]]}
        r=self.assert_reference(p,query=[1,4]);self.assertEqual(r['maximizing_count'],6*6*2)
    def test_subnormal_gap_without_tolerance(self):
        tiny=math.ulp(0.0);r=cyclic_map({'version':1,'fields':[tiny],'edges':[]},query=[0]);self.assertTrue(r['unique']);self.assertEqual(fraction(r['query_states'][0]['gap_to_optimum']),2*F(tiny))
    def test_binary_not_decimal(self):
        r=cyclic_map({'version':1,'fields':[.1],'edges':[]});self.assertEqual(fraction(r['optimal_score']),F(.1));self.assertNotEqual(fraction(r['optimal_score']),F('0.1'))
    def test_forest_agreement(self):
        for n in [1,4,64]:
            p={'version':1,'fields':[.125 if i%2 else 0 for i in range(n)],'edges':[{'i':i,'j':i+1,'weight':-.5} for i in range(n-1)]}
            a=cyclic_map(p);b=forest_map(p);self.assertEqual(a['optimal_score'],b['optimal_score']);self.assertEqual(a['maximizing_count'],b['maximizing_count'])
    def test_caps_equality_and_one_below(self):
        p=model([0]*4,[1]*6);base=elimination_plan(p,query=[0]);limits=[('MAX_BUCKET_STATES','peak_bucket_states'),('MAX_FACTOR_READS','factor_entry_reads'),('MAX_OUTPUT_ENTRIES','total_output_entries')]
        for name,key in limits:
            with patch('varilab.elimination.'+name,base[key]):cyclic_map(p,query=[0])
            with patch('varilab.elimination.'+name,base[key]-1),patch.object(importlib.import_module('varilab.cyclic_map'),'_initial',side_effect=AssertionError('allocated')):
                with self.assertRaisesRegex(ValueError,'before numeric'):cyclic_map(p,query=[0])
    def test_dense_64_preflight(self):
        p=model([0]*64,[1]*2016)
        with patch.object(importlib.import_module('varilab.cyclic_map'),'_initial',side_effect=AssertionError('allocated')):
            with self.assertRaises(ValueError):cyclic_map(p)
    def test_plan_shared_exactly(self):
        p=model([0]*3,[-1]*3);self.assertEqual(cyclic_map(p,query=[2])['plan'],elimination_plan(p,query=[2]))
    def test_zero_edges_preserved(self):
        p=model([0]*3,[0]*3);r=cyclic_map(p);self.assertEqual(r['plan']['peak_bucket_states'],8);self.assertEqual(r['maximizing_count'],8)
    def test_query_canonical_and_empty(self):
        p=model([0]*3,[-1]*3);self.assertEqual(cyclic_map(p,query=[2,0]),cyclic_map(p,query=[0,2]));r=cyclic_map(p);self.assertEqual(len(r['query_states']),1);self.assertEqual(r['query_states'][0]['state'],[])
    def test_relabel_scores_counts(self):
        p=model([.25,-.5,.125],[-1,.5,-.25]);mapping={0:2,1:0,2:1};q={'version':1,'fields':[0]*3,'edges':[]}
        for i,v in enumerate(p['fields']):q['fields'][mapping[i]]=v
        for e in p['edges']:
            a,b=sorted([mapping[e['i']],mapping[e['j']]]);q['edges'].append({'i':a,'j':b,'weight':e['weight']})
        a=cyclic_map(p);b=cyclic_map(q);self.assertEqual(a['optimal_score'],b['optimal_score']);self.assertEqual(a['maximizing_count'],b['maximizing_count'])
    def test_input_output_rng_isolation(self):
        p=model([0]*3,[-1]*3);before=copy.deepcopy(p);random.seed(88);state=random.getstate();a=cyclic_map(p);a['factor_trace'][0]['scores'][0]['numerator']=999
        self.assertEqual(p,before);self.assertEqual(random.getstate(),state);self.assertNotEqual(a,cyclic_map(p))
    def test_deterministic(self):
        p=model([0]*4,[-1]*6);self.assertEqual(cyclic_map(p),cyclic_map(p))
    def test_invalid_query(self):
        p=model([0]*3,[0]*3)
        for q in [True,[True],[0,0],[-1],[3],{},list(range(9))]:
            with self.assertRaises(ValueError):cyclic_map(p,query=q)
    def test_invalid_evidence(self):
        p=model([0]*3,[0]*3)
        for e in [[{'i':True,'spin':1}],[{'i':0,'spin':0}],[{'i':0,'spin':1},{'i':0,'spin':1}]]:
            with self.assertRaises(ValueError):cyclic_map(p,evidence=e)
        with self.assertRaises(ValueError):cyclic_map(p,evidence=[{'i':0,'spin':1}],query=[0])
    def test_invalid_order_and_model(self):
        p=model([0]*3,[0]*3)
        for order in [[0,1],True,[0,1,1],[True,1,2]]:
            with self.assertRaises(ValueError):cyclic_map(p,order=order)
        for p in [{'version':1,'fields':[0]*65,'edges':[]},model([math.inf],[1])]:
            with self.assertRaises(ValueError):cyclic_map(p)
    def test_cli_equal(self):
        p=model([0]*3,[-1]*3)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json';path.write_text(json.dumps(p));out=subprocess.run([sys.executable,'-m','varilab','cyclic-map',str(path),'--query','[2]','--evidence','[{"i":0,"spin":1}]','--order','[1]'],text=True,capture_output=True)
        self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),cyclic_map(p,query=[2],evidence=[{'i':0,'spin':1}],order=[1]))
    def test_cli_demo(self):
        out=subprocess.run([sys.executable,'-m','varilab','cyclic-map-demo'],text=True,capture_output=True);self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout),cyclic_map_demo())
    def test_cli_duplicate_and_error(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json';path.write_text(json.dumps(model([0]*3,[-1]*3)))
            for flag,text in [('--evidence','[{"i":0,"i":1,"spin":1}]'),('--query','[true]'),('--order','[0]')]:
                out=subprocess.run([sys.executable,'-m','varilab','cyclic-map',str(path),flag,text],text=True,capture_output=True);self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'')
