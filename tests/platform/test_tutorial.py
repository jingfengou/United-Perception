"""Keep the standalone tutorial's code tree and examples aligned with source."""
import ast
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Tutorial(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.data = ''
        self.in_data = False
        self.resources = []
        self.ids = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag in ('script', 'iframe', 'img', 'link') and ('src' in attrs or 'href' in attrs):
            self.resources.append(attrs)
        if tag == 'script' and attrs.get('id') == 'tutorial-data':
            self.in_data = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.data += data


class TutorialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT/'docs/platform/TUTORIAL.html').read_text(encoding='utf8')
        cls.document = Tutorial(cls.source)

    def test_tutorial_opens_without_the_chat_host_or_external_resources(self):
        self.assertIn('<!doctype html>', self.source.lower())
        self.assertFalse(self.document.resources)
        self.assertNotIn('window.openai', self.source)
        self.assertEqual(len(self.document.ids), len(set(self.document.ids)))
        self.assertTrue({'architecture', 'runtime-diagram', 'operators-diagram'} <= set(self.document.ids))

    def test_all_fifteen_examples_match_current_interfaces(self):
        data = json.loads(self.document.data)
        categories = {p.parent.name for p in (ROOT/'up_platform/ops').glob('*/base.py')}
        self.assertEqual(set(data['operators']), categories)
        for category, item in data['operators'].items():
            tree = ast.parse((ROOT/f'up_platform/ops/{category}/base.py').read_text(encoding='utf8'))
            methods = [n.name for cls in tree.body if isinstance(cls, ast.ClassDef)
                       for n in cls.body if isinstance(n, ast.FunctionDef)]
            self.assertEqual(item['methods'], methods, category)
            self.assertIn(item['method'], methods, category)
            for vendor in ('nvidia', 'metax'):
                self.assertTrue((ROOT/f'up_platform/ops/{category}/{vendor}.py').is_file())
            self.assertTrue((ROOT/f'up_platform/ops/{category}/{item["impl"]}').is_file())

    def test_code_tree_contains_real_files_and_the_whole_platform(self):
        data = json.loads(self.document.data)
        files = data['files']
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
                      .decode('utf8').rstrip('\0').split('\0'))
        self.assertEqual(len(files), len(set(files)))
        self.assertFalse(set(files) - tracked)
        self.assertFalse({p for p in tracked if p.startswith('up_platform/')} - set(files))
        for path in files:
            self.assertTrue((ROOT/path).is_file(), path)
        self.assertTrue({'up/utils/env/platform.py', 'up_docker', 'docker/unified/launch.py',
                         'tools/new_platform_vendor.py', '.github/workflows/platform.yml'} <= set(files))

    def test_walkthrough_snippets_are_real_current_source(self):
        data = json.loads(self.document.data)
        self.assertTrue({'up_platform', 'up_platform/factory.py', 'up_platform/backends',
                         'up_platform/backends/torch_device.py', 'up_platform/ops/detection',
                         'up_platform/training', 'up_platform/inference', 'docker/unified'}
                        <= set(data['lessons']))
        for path, lesson in data['lessons'].items():
            self.assertTrue((ROOT/path).exists(), path)
            self.assertTrue(all(lesson.get(key) for key in ('job', 'how', 'use')), path)
            for link in lesson.get('links', []):
                self.assertTrue((ROOT/link).exists(), link)
            for snippet in lesson.get('snippets', []):
                source = (ROOT/snippet['path']).read_text(encoding='utf8').splitlines()
                start = snippet['start']-1
                length = len(snippet['code'].splitlines())
                self.assertEqual(snippet['code'], '\n'.join(source[start:start+length]), path)


if __name__ == '__main__':
    unittest.main()
