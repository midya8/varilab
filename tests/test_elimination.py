import copy,itertools,json,math,random,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
from varilab import eliminate,elimination_plan,forest
from varilab.elimination import elimination_demo


def model(n=4,j=.5):return {'version':1,'fields':[0]*n,'edges':[{'i':i,'j':i+1,'weight':j} for i in range(n-1)]}
def oracle(p,query,ev):
    scores=[];states=[]
    for z in itertools.product((-1,1),repeat=len(p['fields'])):
        if any(z[e['i']]!=e['spin'] for e in ev):continue
        value=sum(a*b for a,b in zip(p['fields'],z))+sum(e['weight']*z[e['i']]*z[e['j']] for e in p['edges']);states.append(z);scores.append(math.exp(value))
    norm=math.fsum(scores);probs={q:math.fsum(v for z,v in zip(states,scores) if tuple(z[i] for i in sorted(query))==q)/norm for q in itertools.product((-1,1),repeat=len(query))}
    return math.log(norm),probs


class EliminationTests(unittest.TestCase):
    def check_oracle(self,p,query=(),ev=(),order=None):
        out=eliminate(p,query=query,evidence=ev,order=order);lz,probs=oracle(p,query,ev);self.assertAlmostEqual(out['log_partition'],lz,places=11)
        for row in out['query_states']:
            expected=probs[tuple(row['state'])];self.assertAlmostEqual(row['probability'],expected,places=12);self.assertAlmostEqual(row['log_probability'],math.log(expected),places=11)
        self.assertEqual(out['executed_factor_entry_reads'],out['plan']['factor_entry_reads']);self.assertEqual(out['executed_output_entries'],out['plan']['total_output_entries']);return out
    def test_432_cyclic_conditioned_queries(self):
        for h in itertools.product((-.5,.5),repeat=3):
            for j in (-1,0,1):
                p={'version':1,'fields':list(h),'edges':[{'i':0,'j':1,'weight':j},{'i':1,'j':2,'weight':.25},{'i':0,'j':2,'weight':-.75}]}
                for q,e in (([],[]),([0],[]),([2,0],[]),([0],[{'i':1,'spin':1}]),([],[{'i':0,'spin':-1},{'i':2,'spin':1}]),([1,2],[{'i':0,'spin':-1}])):
                    for order in (None,list(reversed(sorted(set(range(3))-set(q)-{v['i'] for v in e}))),sorted(set(range(3))-set(q)-{v['i'] for v in e})):self.check_oracle(p,q,e,order)
    def test_all_24_orders_dense_four_nodes(self):
        p={'version':1,'fields':[.1,-.3,.2,.8],'edges':[{'i':i,'j':j,'weight':(-1)**(i+j)*.3} for i,j in itertools.combinations(range(4),2)]}
        for order in itertools.permutations(range(4)):self.check_oracle(p,order=order)
    def test_query_canonical_state_order(self):
        out=self.check_oracle(model(),[3,0]);self.assertEqual(out['plan']['query'],[0,3]);self.assertEqual([r['state'] for r in out['query_states']],[[-1,-1],[-1,1],[1,-1],[1,1]])
    def test_all_observed_partition_is_score(self):
        p=model();ev=[{'i':i,'spin':(-1)**i} for i in range(4)];out=self.check_oracle(p,ev=ev);self.assertEqual(out['plan']['steps'],[]);self.assertEqual(out['query_states'],[{'state':[],'log_probability':0.0,'probability':1.0}])
    def test_all_query_no_elimination(self):
        out=self.check_oracle(model(3),[0,1,2]);self.assertEqual(out['plan']['steps'],[]);self.assertEqual(out['plan']['final_entries'],8)
    def test_isolated_and_disconnected(self):
        p=model(6);p['fields']=[.1,0,-.5,.8,-.2,.3];p['edges']=[p['edges'][0],p['edges'][3]];self.check_oracle(p,[0,2,5],[{'i':4,'spin':-1}])
    def test_single_node(self):
        p={'version':1,'fields':[.7],'edges':[]};self.check_oracle(p);self.check_oracle(p,[0]);self.check_oracle(p,ev=[{'i':0,'spin':1}])
    def test_star_rejected_order_not_rejected_graph(self):
        d=elimination_demo();a=d['hub_first_plan'];b=d['min_fill_star']['plan'];self.assertFalse(a['accepted']);self.assertTrue(b['accepted']);self.assertEqual(a['peak_bucket_states'],65536);self.assertEqual(b['peak_bucket_states'],4)
    def test_rejection_precedes_numeric_table_creation(self):
        p={'version':1,'fields':[0]*16,'edges':[{'i':0,'j':i,'weight':.5} for i in range(1,16)]}
        with patch('varilab.elimination._initial_factors',side_effect=AssertionError('numeric allocation')):
            plan=elimination_plan(p,order=list(range(16)));self.assertFalse(plan['accepted'])
            with self.assertRaisesRegex(ValueError,'before numeric'):eliminate(p,order=list(range(16)))
    def test_planner_is_nonnumeric_even_accepted(self):
        with patch('varilab.elimination._initial_factors',side_effect=AssertionError):self.assertTrue(elimination_plan(model())['accepted'])
    def test_resource_exact_boundaries(self):
        p=model();base=elimination_plan(p)
        for key,field in [('MAX_BUCKET_STATES','peak_bucket_states'),('MAX_FACTOR_READS','factor_entry_reads'),('MAX_OUTPUT_ENTRIES','total_output_entries')]:
            with patch('varilab.elimination.'+key,base[field]):self.assertTrue(elimination_plan(p)['accepted']);eliminate(p)
            with patch('varilab.elimination.'+key,base[field]-1):self.assertFalse(elimination_plan(p)['accepted'])
    def test_dense_64_plan_symbolic_only(self):
        p={'version':1,'fields':[0]*64,'edges':[{'i':i,'j':j,'weight':0} for i,j in itertools.combinations(range(64),2)]}
        with patch('varilab.elimination._initial_factors',side_effect=AssertionError):out=elimination_plan(p,order=list(range(64)))
        self.assertEqual(out['peak_bucket_states'],2**64);self.assertFalse(out['accepted']);self.assertEqual(len(out['steps']),64)
    def test_64_cycle_analytic(self):
        out=elimination_demo()['cycle_query'];self.assertEqual(len(out['plan']['order']),62);self.assertEqual(out['plan']['peak_bucket_states'],8);self.assertAlmostEqual(sum(r['probability'] for r in out['query_states']),1)
    def test_factor_trace_source_scopes_and_lengths(self):
        out=eliminate(model(8),query=[1,6]);scopes={f'u{i}':{i} for i in range(8)};scopes.update({f'p{i}':{i,i+1} for i in range(7)})
        for step,trace in zip(out['plan']['steps'],out['factor_trace']):
            bag=set().union(*(scopes[k] for k in step['input_factor_ids']));self.assertEqual(sorted(bag),step['bag']);self.assertEqual(step['assignment_count'],2**len(bag));self.assertEqual(step['factor_entry_reads'],2**len(bag)*len(step['input_factor_ids']))
            for k in step['input_factor_ids']:del scopes[k]
            scope=bag-{step['eliminate']};scopes[step['output_factor_id']]=scope;self.assertEqual(trace['scope'],sorted(scope));self.assertEqual(len(trace['log_values']),2**len(scope))
        self.assertEqual(list(scopes),out['plan']['final_factor_ids'])
    def test_independent_trace_numeric_factor(self):
        p=model(3);p['fields']=[.3,-.2,.1];out=eliminate(p,order=[0,1,2]);first=out['factor_trace'][0]
        self.assertEqual(first['scope'],[1])
        for y,score in zip((-1,1),first['log_values']):self.assertAlmostEqual(score,math.log(sum(math.exp(.3*x+.5*x*y) for x in (-1,1))),places=12)
    def test_minfill_determinism_and_edge_order(self):
        p=model(10);a=eliminate(p,query=[4]);p['edges'].reverse();self.assertEqual(a,eliminate(p,query=[4]))
    def test_minfill_ties_choose_degree_then_index(self):
        p=model(4);out=elimination_plan(p);self.assertEqual(out['order'],[0,1,2,3])
    def test_underflow_possible_query_retains_log(self):
        p={'version':1,'fields':[100]*8,'edges':[]};out=eliminate(p,query=list(range(8)));row=out['query_states'][0];self.assertEqual(row['probability'],0);self.assertTrue(math.isfinite(row['log_probability']));self.assertEqual(row['log_probability'],-1600)
    def test_zero_edges_kept_in_symbolic_scopes(self):
        p={'version':1,'fields':[0]*16,'edges':[{'i':0,'j':i,'weight':0} for i in range(1,16)]};self.assertFalse(elimination_plan(p,order=list(range(16)))['accepted'])
    def test_no_evidence_probability_claim(self):
        p=model(2);out=eliminate(p,evidence=[{'i':0,'spin':1}]);self.assertNotIn('evidence_probability',out);self.assertAlmostEqual(out['log_partition'],math.log(2*math.cosh(.5)))
    def test_old_forest_partition_matches(self):
        p=model(12);p['fields']=[i/20 for i in range(12)];ev=[{'i':3,'spin':-1}];self.assertAlmostEqual(eliminate(p,evidence=ev)['log_partition'],forest(p,evidence=ev)['conditional']['log_partition'],places=11)
    def test_query_validation(self):
        for q in ({},[True],[4],[0,0],list(range(9))):
            with self.assertRaises(ValueError):eliminate(model(),query=q)
        with self.assertRaises(ValueError):eliminate(model(),query=[0],evidence=[{'i':0,'spin':1}])
    def test_order_validation(self):
        for order in ([0,1,2],[0,1,2,True],[0,1,1,3],{},[0,1,2,4]):
            with self.assertRaises(ValueError):elimination_plan(model(),order=order)
        with self.assertRaises(ValueError):elimination_plan(model(),query=[0],order=[0,1,2,3])
    def test_model_and_size_validation(self):
        for p in (model(0),model(65),{'version':1,'fields':[float('inf')],'edges':[]},{'version':1,'fields':[101],'edges':[]}):
            with self.assertRaises(ValueError):eliminate(p)
    def test_evidence_validation(self):
        for ev in ([{'i':True,'spin':1}],[{'i':0,'spin':True}],[{'i':0,'spin':1}]*2,[{'i':4,'spin':1}],[{'i':0,'spin':1,'x':0}]):
            with self.assertRaises(ValueError):eliminate(model(),evidence=ev)
    def test_input_output_rng_isolation(self):
        p=model();q=[3,0];e=[{'i':1,'spin':1}];snap=copy.deepcopy((p,q,e));rng=random.getstate();out=eliminate(p,query=q,evidence=e);out['factor_trace'][0]['log_values'][0]=999;out['plan']['steps'][0]['bag'].append(999);self.assertEqual((p,q,e),snap);self.assertEqual(random.getstate(),rng);self.assertNotEqual(out,eliminate(p,query=q,evidence=e))
    def test_cli_api_and_plan(self):
        p=json.loads(Path('examples/coupled.json').read_text())
        for cmd,fn in [('eliminate',eliminate),('plan',elimination_plan)]:
            out=subprocess.run([sys.executable,'-m','varilab',cmd,'examples/coupled.json','--query','[1]','--evidence','[{"i":0,"spin":1}]'],capture_output=True,text=True,check=True);self.assertEqual(json.loads(out.stdout),fn(p,query=[1],evidence=[{'i':0,'spin':1}]))
    def test_cli_demo(self):
        out=subprocess.run([sys.executable,'-m','varilab','elimination-demo'],capture_output=True,text=True,check=True);self.assertEqual(json.loads(out.stdout),elimination_demo())
    def test_cli_invalid_query_and_duplicate_evidence(self):
        for flags in (['--query','[true]'],['--evidence','[{"i":0,"i":1,"spin":1}]']):
            out=subprocess.run([sys.executable,'-m','varilab','eliminate','examples/coupled.json',*flags],capture_output=True,text=True);self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertIn('error:',out.stderr)
    def test_demo_gate_directions(self):self.assertEqual(elimination_demo()['gates'],{'hub_first_plan_accepted':False,'min_fill_star_plan_accepted':True,'cycle_partition_matches_analytic':True})
