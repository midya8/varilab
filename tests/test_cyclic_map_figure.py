import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from examples.render_cyclic_map import render
from varilab.cyclic_map import cyclic_map_demo
NS={'s':'http://www.w3.org/2000/svg'}


class CyclicMapFigureTests(unittest.TestCase):
    def test_signed_score_geometry(self):
        root=ET.fromstring(render(cyclic_map_demo()));bars=[r for r in root.findall('s:rect',NS) if 'data-score' in r.attrib];self.assertEqual(len(bars),2)
        for bar,v in zip(bars,[-3,1]):
            self.assertEqual(int(bar.attrib['data-score']),v);self.assertEqual(int(bar.attrib['y']),340-50*max(v,0));self.assertEqual(int(bar.attrib['height']),50*abs(v));self.assertEqual(int(bar.attrib['width']),100)
        self.assertTrue(any(p.attrib['d']=='M 80 340 H 610' and p.attrib['stroke-width']=='2' for p in root.findall('s:path',NS)))
    def test_complete_exact_metadata(self):
        d=cyclic_map_demo();root=ET.fromstring(render(d));self.assertEqual(json.loads(root.find('s:metadata',NS).text),d)
    def test_asset_regeneration(self):
        self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'cyclic-map-audit.svg').read_text(),render(cyclic_map_demo()))
    def test_reject_changed_trace(self):
        d=copy.deepcopy(cyclic_map_demo());d['triangle']['factor_trace'][0]['maximizing_counts'][0]+=1
        with self.assertRaises(ValueError):render(d)
