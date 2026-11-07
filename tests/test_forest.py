import copy,itertools,json,math,random,subprocess,sys,unittest
from unittest.mock import patch
from varilab import forest,exact,mean_field
from varilab.forest import forest_demo


def model(n=4,j=2):return {'version':1,'fields':[0]*n,'edges':[{'i':i,'j':i+1,'weight':j} for i in range(n-1)]}
def oracle(p,evidence=()):
    states=list(itertools.product((-1,1),repeat=len(p['fields'])))
    values=[math.exp(math.fsum([a*b for a,b in zip(p['fields'],z)]+[e['weight']*z[e['i']]*z[e['j']] for e in p['edges']])) for z in states]
    total=math.fsum(values);restricted=[v if all(z[e['i']]==e['spin'] for e in evidence) else 0.0 for z,v in zip(states,values)];norm=math.fsum(restricted)
    return states,[v/norm for v in restricted],math.log(norm),norm/total


class ForestTests(unittest.TestCase):
    def check_oracle(self,p,evidence=()):
        out=forest(p,evidence=evidence);states,probs,lz,ep=oracle(p,evidence)
        self.assertAlmostEqual(out['conditional']['log_partition'],lz,places=11)
        self.assertAlmostEqual(out['evidence_probability'],ep,places=11)
        for row in out['conditional']['singletons']:
            i=row['i'];expected=[math.fsum(a for a,z in zip(probs,states) if z[i]==v) for v in (-1,1)]
            for a,b in zip(row['probabilities'],expected):self.assertAlmostEqual(a,b,places=11)
        for edge in out['conditional']['edge_joints']:
            for row in edge['states']:
                expected=math.fsum(a for a,z in zip(probs,states) if [z[edge['i']],z[edge['j']]]==row['state'])
                self.assertAlmostEqual(row['probability'],expected,places=11)
        return out
    def test_324_conditioned_tree_oracles(self):
        for h in itertools.product((-.5,0,.5),repeat=3):
            for j in (-1.5,0,1.5):
                p={'version':1,'fields':list(h),'edges':[{'i':0,'j':1,'weight':j},{'i':0,'j':2,'weight':-.7}]}
                for e in ([],[{'i':0,'spin':1}],[{'i':1,'spin':-1}],[{'i':0,'spin':1},{'i':2,'spin':-1}]):self.check_oracle(p,e)
    def test_disconnected_components(self):
        p=model(6);p['edges']=[p['edges'][0],p['edges'][3]];out=self.check_oracle(p,[{'i':4,'spin':-1}]);self.assertEqual(out['roots'],[0,2,3,5])
    def test_isolated_nodes(self):
        p={'version':1,'fields':[.3,-.8,1.2],'edges':[]};out=self.check_oracle(p,[{'i':0,'spin':1}]);self.assertEqual(out['conditional']['message_count'],0)
        self.assertAlmostEqual(out['conditional']['singletons'][1]['mean'],math.tanh(-.8))
    def test_single_node(self):
        for z in (-1,1):self.check_oracle(model(1),[{'i':0,'spin':z}])
    def test_all_observed(self):
        p=model();e=[{'i':i,'spin':(-1)**i} for i in range(4)];out=self.check_oracle(p,e)
        self.assertTrue(all(row['probabilities'].count(0)==1 for row in out['conditional']['singletons']))
    def test_chain_analytic_means(self):
        out=forest(model(12),evidence=[{'i':0,'spin':1}])
        for i,row in enumerate(out['conditional']['singletons']):self.assertAlmostEqual(row['mean'],math.tanh(2)**i,places=12)
    def test_128_chain_no_recursion_or_enumeration(self):
        with patch('varilab.core.exact',side_effect=AssertionError),patch('varilab.core.mean_field',side_effect=AssertionError):out=forest(model(128),evidence=[{'i':0,'spin':1}])
        self.assertEqual(len(out['conditional']['singletons']),128);self.assertEqual(out['conditional']['message_count'],254)
        self.assertAlmostEqual(out['unconditional_log_partition'],math.log(2)+127*math.log(2*math.cosh(2)),places=9)
    def test_star_analytic_partition(self):
        p={'version':1,'fields':[0]*128,'edges':[{'i':0,'j':i,'weight':.2} for i in range(1,128)]};out=forest(p)
        self.assertAlmostEqual(out['conditional']['log_partition'],math.log(2)+127*math.log(2*math.cosh(.2)),places=10)
    def test_directed_message_subtree_oracle(self):
        p={'version':1,'fields':[.3,-.2,.5,-.1],'edges':[{'i':0,'j':1,'weight':.8},{'i':1,'j':2,'weight':-.7},{'i':1,'j':3,'weight':.2}]};ev={2:-1};out=forest(p,evidence=[{'i':2,'spin':-1}])
        for row in out['conditional']['directed_messages']:
            i,j=row['from'],row['to'];edges=[e for e in p['edges'] if {e['i'],e['j']}!={i,j}];side={i}
            for _ in range(4):
                for e in edges:
                    if e['i'] in side or e['j'] in side:side.update([e['i'],e['j']])
            ids=sorted(side);w=next(e['weight'] for e in p['edges'] if {e['i'],e['j']}=={i,j})
            for s,y in enumerate((-1,1)):
                terms=[]
                for values in itertools.product((-1,1),repeat=len(ids)):
                    z=dict(zip(ids,values))
                    if any(k in z and z[k]!=v for k,v in ev.items()):continue
                    energy=sum(p['fields'][k]*z[k] for k in ids)+sum(e['weight']*z[e['i']]*z[e['j']] for e in edges if e['i'] in side)+w*z[i]*y
                    terms.append(math.exp(energy))
                self.assertAlmostEqual(row['log_values'][s],math.log(sum(terms)),places=12)
    def test_message_dependency_schedule(self):
        out=forest(model(8));seen=set()
        for row in out['conditional']['directed_messages']:
            for k in row['depends_on']:self.assertIn((k,row['from']),seen)
            self.assertNotIn((row['from'],row['to']),seen);seen.add((row['from'],row['to']))
        self.assertEqual(len(seen),14)
    def test_edge_marginal_consistency(self):
        out=forest(model(7),evidence=[{'i':2,'spin':-1}])['conditional']
        for edge in out['edge_joints']:
            for k,i in enumerate((edge['i'],edge['j'])):
                for s,z in enumerate((-1,1)):self.assertAlmostEqual(sum(row['probability'] for row in edge['states'] if row['state'][k]==z),out['singletons'][i]['probabilities'][s],places=12)
    def test_rare_evidence_keeps_finite_logs(self):
        p={'version':1,'fields':[100]*128,'edges':[]};out=forest(p,evidence=[{'i':i,'spin':-1} for i in range(128)])
        self.assertEqual(out['evidence_probability'],0);self.assertTrue(math.isfinite(out['log_evidence_probability']));self.assertAlmostEqual(out['log_evidence_probability'],-25600)
        json.dumps(out,allow_nan=False)
    def test_underflow_not_forbidden(self):
        p=model(8,100);p['fields']=[100]*8;out=forest(p,evidence=[{'i':7,'spin':-1}]);json.dumps(out,allow_nan=False)
        self.assertTrue(all(v is not None for row in out['conditional']['singletons'][:7] for v in row['log_probabilities']))
    def test_empty_evidence_probability_one(self):
        out=forest(model());self.assertEqual(out['evidence_probability'],1);self.assertEqual(out['log_evidence_probability'],0)
    def test_cycle_rejection_before_messages(self):
        p=model(5);p['edges'].append({'i':0,'j':2,'weight':0})
        import importlib
        with patch.object(importlib.import_module('varilab.forest'),'_pass',side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError,'acyclic'):forest(p)
    def test_cycle_when_edgecount_within_forest_bound(self):
        p={'version':1,'fields':[0]*5,'edges':[{'i':0,'j':1,'weight':1},{'i':0,'j':2,'weight':1},{'i':1,'j':2,'weight':1}]}
        with self.assertRaisesRegex(ValueError,'acyclic'):forest(p)
    def test_zero_edges_are_stored(self):
        out=forest(model(3,0));self.assertEqual(out['conditional']['message_count'],4);self.assertEqual(len(out['conditional']['edge_joints']),2)
    def test_forest_size_bound_existing_bounds_unchanged(self):
        with self.assertRaises(ValueError):forest(model(129))
        for fn in (exact,mean_field):
            with self.assertRaises(ValueError):fn(model(13))
    def test_evidence_validation(self):
        for ev in ({},'x',[{}],[{'i':0,'spin':True}],[{'i':True,'spin':1}],[{'i':-1,'spin':1}],[{'i':4,'spin':1}],[{'i':0,'spin':0}],[{'i':0,'spin':1},{'i':0,'spin':-1}],[{'i':0,'spin':1,'extra':0}]):
            with self.assertRaises(ValueError):forest(model(),evidence=ev)
    def test_model_validation(self):
        for p in ({},dict(model(),version=True),dict(model(),fields=[float('inf')]),dict(model(),edges=[{'i':1,'j':0,'weight':1}])):
            with self.assertRaises(ValueError):forest(p)
    def test_input_output_rng_isolation(self):
        p=model();e=[{'i':1,'spin':1}];before=copy.deepcopy((p,e));rng=random.getstate();out=forest(p,evidence=e)
        out['model']['fields'][0]=9;out['evidence'][0]['spin']=-1;self.assertEqual((p,e),before);self.assertEqual(random.getstate(),rng)
    def test_permutation_equivariance(self):
        p=model();old=forest(p,evidence=[{'i':0,'spin':1}]);perm=[3,1,0,2]
        newedges=[]
        for e in p['edges']:
            i,j=sorted([perm.index(e['i']),perm.index(e['j'])]);newedges.append({'i':i,'j':j,'weight':e['weight']})
        out=forest({'version':1,'fields':[0]*4,'edges':newedges},evidence=[{'i':perm.index(0),'spin':1}])
        for i,j in enumerate(perm):self.assertAlmostEqual(out['conditional']['singletons'][i]['mean'],old['conditional']['singletons'][j]['mean'],places=12)
    def test_demo_both_query_gates(self):
        out=forest_demo();self.assertTrue(out['gates']['conditional_endpoint_mean_gt_point8']);self.assertFalse(out['gates']['unconditioned_endpoint_mean_gt_point8'])
    def test_cli_matches_api(self):
        run=subprocess.run([sys.executable,'-m','varilab','forest','examples/coupled.json','--evidence','[{"i":0,"spin":1}]'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout),forest(model(2,3),evidence=[{'i':0,'spin':1}]))
    def test_cli_bad_evidence_no_stdout(self):
        run=subprocess.run([sys.executable,'-m','varilab','forest','examples/coupled.json','--evidence','[{"i":0,"spin":0}]'],capture_output=True,text=True)
        self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'');self.assertIn('spin',run.stderr)
