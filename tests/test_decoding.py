import copy,itertools,json,math,random,subprocess,sys,unittest
from fractions import Fraction as F
from unittest.mock import patch
from varilab import forest_map,forest
from varilab.decoding import map_demo


def val(p):return None if p is None else F(p['numerator'],p['denominator'])
def model(n=4,j=2):return {'version':1,'fields':[0]*n,'edges':[{'i':i,'j':i+1,'weight':j} for i in range(n-1)]}
def energy(p,z):return sum(F(float(h))*s for h,s in zip(p['fields'],z))+sum(F(float(e['weight']))*z[e['i']]*z[e['j']] for e in p['edges'])
def oracle(p,ev=()):
    return [(z,energy(p,z)) for z in itertools.product((-1,1),repeat=len(p['fields'])) if all(z[e['i']]==e['spin'] for e in ev)]


class DecodingTests(unittest.TestCase):
    def check_oracle(self,p,ev=()):
        out=forest_map(p,evidence=ev);states=oracle(p,ev);score=max(s for z,s in states);winners=[z for z,s in states if s==score]
        self.assertEqual(val(out['optimal_score']),score);self.assertEqual(val(out['assignment_score']),score)
        self.assertIn(tuple(out['assignment']),winners);self.assertEqual(out['maximizing_count'],len(winners));self.assertEqual(out['unique'],len(winners)==1)
        self.assertEqual(energy(p,out['assignment']),score)
        for row in out['max_marginals']:
            for r in row['states']:
                restricted=[s for z,s in states if z[row['i']]==r['spin']]
                expected=max(restricted) if restricted else None
                self.assertEqual(val(r['score']),expected)
                self.assertEqual(r['maximizing_count'],restricted.count(expected))
                self.assertEqual(val(r['gap_to_optimum']),score-expected if expected is not None else None)
                self.assertEqual(r['attains_global_optimum'],expected==score)
        return out
    def test_972_exhaustive_conditioned_cases(self):
        for h in itertools.product((-.5,0,.5),repeat=3):
            for a,b in itertools.product((-1,0,1),repeat=2):
                p={'version':1,'fields':list(h),'edges':[{'i':0,'j':1,'weight':a},{'i':0,'j':2,'weight':b}]}
                for ev in ([],[{'i':0,'spin':1}],[{'i':1,'spin':-1}],[{'i':0,'spin':1},{'i':2,'spin':-1}]):self.check_oracle(p,ev)
    def test_branched_tree_and_all_evidence(self):
        p={'version':1,'fields':[.1,-.7,0,.8,-.2,.5],'edges':[{'i':0,'j':4,'weight':1},{'i':1,'j':4,'weight':-.2},{'i':2,'j':4,'weight':.1},{'i':2,'j':3,'weight':.3},{'i':3,'j':5,'weight':-.9}]}
        self.check_oracle(p,[{'i':2,'spin':-1}]);out=self.check_oracle(p,[{'i':i,'spin':(-1)**i} for i in range(6)]);self.assertEqual(out['maximizing_count'],1)
    def test_disconnected_global_max_marginal_scores_counts(self):
        p=model(6);p['edges']=[p['edges'][0],p['edges'][3]];self.check_oracle(p);out=self.check_oracle(p,[{'i':4,'spin':1}]);self.assertEqual(out['roots'],[0,2,3,5])
    def test_single_spin(self):
        for h in (0,.1,-.1):
            p={'version':1,'fields':[h],'edges':[]};self.check_oracle(p)
            for z in (-1,1):self.check_oracle(p,[{'i':0,'spin':z}])
    def test_128_independent_exact_counts(self):
        p=model(128);p['edges']=[];out=forest_map(p);self.assertEqual(out['maximizing_count'],2**128);self.assertEqual(out['assignment'],[-1]*128)
        for row in out['max_marginals']:
            self.assertEqual([s['maximizing_count'] for s in row['states']],[2**127,2**127])
    def test_128_chain_analytic(self):
        with patch('varilab.core.exact',side_effect=AssertionError):out=forest_map(model(128))
        self.assertEqual(val(out['optimal_score']),254);self.assertEqual(out['maximizing_count'],2);self.assertEqual(out['message_count'],254);self.assertEqual(out['assignment'],[-1]*128)
    def test_128_star_analytic(self):
        p=model(128);p['edges']=[{'i':0,'j':i,'weight':-2} for i in range(1,128)];out=forest_map(p,evidence=[{'i':0,'spin':1}]);self.assertEqual(val(out['optimal_score']),254);self.assertEqual(out['assignment'],[1]+[-1]*127);self.assertEqual(out['maximizing_count'],1)
    def test_exact_binary_float_not_decimal(self):
        out=self.check_oracle({'version':1,'fields':[.1],'edges':[]});self.assertEqual(val(out['optimal_score']),F(.1));self.assertNotEqual(val(out['optimal_score']),F(1,10))
    def test_subnormal_no_tie_tolerance(self):
        tiny=math.nextafter(0,1);out=self.check_oracle({'version':1,'fields':[tiny],'edges':[]});self.assertTrue(out['unique']);self.assertEqual(out['assignment'],[1]);self.assertEqual(val(out['max_marginals'][0]['states'][0]['gap_to_optimum']),2*F(tiny))
    def test_directed_source_side_exhaustive_oracle(self):
        p={'version':1,'fields':[.3,-.2,.5,-.1],'edges':[{'i':0,'j':1,'weight':.8},{'i':1,'j':2,'weight':-.7},{'i':1,'j':3,'weight':.2}]};ev={2:-1};out=forest_map(p,evidence=[{'i':2,'spin':-1}])
        for row in out['directed_messages']:
            i,j=row['from'],row['to'];edges=[e for e in p['edges'] if {e['i'],e['j']}!={i,j}];side={i}
            for _ in range(4):
                for e in edges:
                    if e['i'] in side or e['j'] in side:side.update([e['i'],e['j']])
            ids=sorted(side);w=F(next(e['weight'] for e in p['edges'] if {e['i'],e['j']}=={i,j}))
            for s,y in enumerate((-1,1)):
                records=[]
                for values in itertools.product((-1,1),repeat=len(ids)):
                    z=dict(zip(ids,values))
                    if any(k in z and z[k]!=v for k,v in ev.items()):continue
                    score=sum(F(p['fields'][k])*z[k] for k in ids)+sum(F(e['weight'])*z[e['i']]*z[e['j']] for e in edges if e['i'] in side)+w*z[i]*y;records.append((z[i],score))
                best=max(v for z,v in records);wins=[z for z,v in records if v==best]
                self.assertEqual(val(row['scores'][s]),best);self.assertEqual(row['maximizing_counts'][s],len(wins));self.assertEqual(row['maximizing_source_spins'][s],sorted(set(wins)))
    def test_dependency_schedule(self):
        seen=set();out=forest_map(model(9))
        for r in out['directed_messages']:
            self.assertTrue(all((k,r['from']) in seen for k in r['depends_on']));self.assertNotIn((r['from'],r['to']),seen);seen.add((r['from'],r['to']))
        self.assertEqual(len(seen),16)
    def test_traceback_compatibility(self):
        out=forest_map(model(8,j=-2));seen=set()
        for r in out['backtrack']:
            if r['parent'] is not None:self.assertIn(r['parent'],seen);self.assertEqual(r['parent_spin'],out['assignment'][r['parent']])
            self.assertEqual(r['chosen_spin'],r['compatible_spins'][0]);self.assertEqual(r['chosen_spin'],out['assignment'][r['i']]);seen.add(r['i'])
        self.assertEqual(len(seen),8)
    def test_tied_local_modes_are_incompatible(self):
        out=self.check_oracle(model(2,j=-2));self.assertEqual(out['assignment'],[-1,1]);self.assertEqual(out['maximizing_count'],2)
        self.assertTrue(all(s['attains_global_optimum'] for row in out['max_marginals'] for s in row['states']));self.assertLess(energy(model(2,j=-2),[-1,-1]),val(out['optimal_score']))
    def test_unique_marginal_mode_counterexample(self):
        out=map_demo();self.assertEqual(out['marginal_modes'],[1,1]);self.assertEqual(out['unique_case']['assignment'],[1,-1]);self.assertTrue(all(p[1]>p[0] for p in out['marginal_probabilities']));self.assertEqual(val(out['score_gap']),F(449,128));self.assertEqual(out['gates'],{'decoded_is_optimal':True,'marginal_modes_are_optimal':False})
    def test_edge_order_determinism(self):
        p=model(7,j=-2);a=forest_map(p);p['edges'].reverse();self.assertEqual(a,forest_map(p))
    def test_relabel_score_count_not_tie_witness(self):
        p=model(5,j=-2);p['fields']=[.5,0,-.5,0,.25];perm=[3,1,4,0,2];q={'version':1,'fields':[p['fields'][i] for i in perm],'edges':[{'i':min(perm.index(e['i']),perm.index(e['j'])),'j':max(perm.index(e['i']),perm.index(e['j'])),'weight':e['weight']} for e in p['edges']]};a=self.check_oracle(p);b=self.check_oracle(q);self.assertEqual(a['optimal_score'],b['optimal_score']);self.assertEqual(a['maximizing_count'],b['maximizing_count'])
    def test_input_output_rng_isolation(self):
        p=model();ev=[{'i':0,'spin':1}];snap=copy.deepcopy((p,ev));rng=random.getstate();out=forest_map(p,evidence=ev);out['assignment'][0]=8;out['directed_messages'][0]['scores'][0]['numerator']=999;self.assertEqual((p,ev),snap);self.assertEqual(random.getstate(),rng);self.assertEqual(forest_map(p,evidence=ev)['assignment'][0],1)
    def test_cycles_including_zero_fail_before_messages(self):
        p=model(4);p['edges']=[{'i':0,'j':1,'weight':0},{'i':1,'j':2,'weight':0},{'i':0,'j':2,'weight':0}]
        with self.assertRaises(ValueError):forest_map(p)
    def test_bounds(self):
        for n in (0,129):
            with self.assertRaises(ValueError):forest_map(model(n))
    def test_evidence_validation(self):
        for ev in ({},[{'i':True,'spin':1}],[{'i':0,'spin':True}],[{'i':0,'spin':0}],[{'i':0,'spin':1}]*2,[{'i':4,'spin':1}],[{'i':0,'spin':1,'x':0}]):
            with self.assertRaises(ValueError):forest_map(model(),evidence=ev)
    def test_model_validation(self):
        for p in ({'version':2,'fields':[0],'edges':[]},{'version':1,'fields':[float('nan')],'edges':[]},{'version':1,'fields':[101],'edges':[]}):
            with self.assertRaises(ValueError):forest_map(p)
    def test_json_round_trip_exact(self):
        out=forest_map(model(128));self.assertEqual(json.loads(json.dumps(out,allow_nan=False)),out)
    def test_cli_matches_api(self):
        out=subprocess.run([sys.executable,'-m','varilab','map','examples/coupled.json','--evidence','[{"i":0,"spin":1}]'],capture_output=True,text=True,check=True)
        from pathlib import Path
        p=json.loads(Path('examples/coupled.json').read_text());self.assertEqual(json.loads(out.stdout),forest_map(p,evidence=[{'i':0,'spin':1}]))
    def test_cli_demo(self):
        out=subprocess.run([sys.executable,'-m','varilab','map-demo'],capture_output=True,text=True,check=True);self.assertEqual(json.loads(out.stdout),map_demo())
    def test_cli_invalid_evidence_no_partial_result(self):
        out=subprocess.run([sys.executable,'-m','varilab','map','examples/coupled.json','--evidence','[{"i":0,"spin":0}]'],capture_output=True,text=True);self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertIn('error:',out.stderr)
    def test_cli_duplicate_evidence_keys_rejected(self):
        for command in ('map','forest'):
            out=subprocess.run([sys.executable,'-m','varilab',command,'examples/coupled.json','--evidence','[{"i":0,"i":1,"spin":1}]'],capture_output=True,text=True)
            self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertIn('duplicate JSON key',out.stderr)
    def test_old_sum_product_unchanged(self):
        p=model(8);a=forest(p,evidence=[{'i':0,'spin':1}]);forest_map(p,evidence=[{'i':0,'spin':1}]);self.assertEqual(a,forest(p,evidence=[{'i':0,'spin':1}]))
