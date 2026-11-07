import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from varilab.demo import demo

spec=importlib.util.spec_from_file_location('render_audit',Path(__file__).resolve().parents[1]/'examples'/'render_audit.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
NS={'s':'http://www.w3.org/2000/svg'}


class FigureTests(unittest.TestCase):
    def test_every_joint_probability_bar(self):
        p=demo();root=ET.fromstring(renderer.render(p));bars=[r for r in root.findall('s:rect',NS) if 'data-state' in r.attrib];self.assertEqual(len(bars),16)
        for r in bars:
            a=r.attrib;j=int(a['data-state']);i=int(a['data-arm'])
            c=p['coupled_arms'][0 if i==0 else i-1]['comparison'];value=c['exact']['states'][j]['probability'] if i==0 else c['q_states'][j]['q_probability']
            self.assertEqual(float(a['data-value']),value);self.assertEqual(float(a['x']),112+245*j+34*i);self.assertEqual(float(a['y']),390-230*value);self.assertEqual(float(a['height']),230*value)
        self.assertEqual(root.attrib['viewBox'],'0 0 1120 740')

    def test_all_table_metrics(self):
        p=demo();root=ET.fromstring(renderer.render(p));records=root.findall('s:g',NS);self.assertEqual(len(records),9)
        for r in records:
            a=r.attrib;c=p['coupled_arms'][int(a['data-row'])]['comparison'];value={'elbo':c['mean_field']['elbo'],'reverse_kl':c['reverse_kl_direct'],'mean_error':c['max_mean_error']}[a['data-metric']]
            self.assertEqual(float(a['data-value']),value)
        text=''.join(root.itertext());self.assertIn('not a maximum',text);self.assertIn('gate fails',text)

    def test_complete_metadata_reproducible_no_mutation(self):
        p=demo();saved=copy.deepcopy(p);svg=renderer.render(p);self.assertEqual(p,saved);self.assertEqual(svg,renderer.render(p))
        self.assertEqual(json.loads(ET.fromstring(svg).find('s:metadata',NS).text),p)
        self.assertEqual(svg,(Path(__file__).resolve().parents[1]/'assets'/'variational-audit.svg').read_text())

    def test_changed_inputs_rejected(self):
        for what in ('model','history','gate','selection'):
            p=demo()
            if what=='model':p['coupled_arms'][0]['comparison']['exact']['model']['fields'][0]=.1
            elif what=='history':p['coupled_arms'][0]['comparison']['mean_field']['history'][0]['elbo']=0
            elif what=='gate':p['gates']['positive_mean_error_le_1e_6']=True
            else:p['selection']='unknown'
            with self.assertRaises(ValueError):renderer.render(p)
