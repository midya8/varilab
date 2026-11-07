import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from examples.render_elimination import render
from varilab.elimination import elimination_demo
NS={'s':'http://www.w3.org/2000/svg'}


class EliminationFigureTests(unittest.TestCase):
    def test_all_logarithmic_bucket_bars_and_cap(self):
        root=ET.fromstring(render(elimination_demo()));rows=[r for r in root.findall('s:rect',NS) if 'data-plan' in r.attrib];self.assertEqual(len(rows),3)
        for row,count in zip(rows,(65536,4,8)):
            exponent=count.bit_length()-1;self.assertEqual(int(row.attrib['data-states']),count);self.assertEqual(int(row.attrib['height']),18*exponent);self.assertEqual(int(row.attrib['y']),450-18*exponent)
        cap=[p for p in root.findall('s:path',NS) if 'data-cap' in p.attrib];self.assertEqual(len(cap),1);self.assertEqual(cap[0].attrib['d'],'M 80 234 H 1070');self.assertEqual(cap[0].attrib['data-cap'],'4096')
    def test_full_symbolic_plans_and_portable_labels(self):
        d=elimination_demo();root=ET.fromstring(render(d));m=json.loads(root.find('s:metadata',NS).text);self.assertEqual(m['plans'],[d['hub_first_plan'],d['min_fill_star']['plan'],d['cycle_query']['plan']]);self.assertEqual(m['gates'],d['gates']);self.assertEqual(m['selection'],d['selection']);self.assertEqual(m['cycle_query_probabilities'],[round(r['probability'],12) for r in d['cycle_query']['query_states']])
    def test_regeneration(self):self.assertEqual(Path('assets/elimination-audit.svg').read_text(),render(elimination_demo()))
    def test_reject_changed_plan(self):
        d=copy.deepcopy(elimination_demo());d['hub_first_plan']['accepted']=True
        with self.assertRaises(ValueError):render(d)
