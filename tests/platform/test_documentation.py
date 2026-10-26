"""Published docs must link to tracked files, not local experiment artifacts."""
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / 'tools/check_platform_docs.py'
spec = importlib.util.spec_from_file_location('platform_docs', SOURCE)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DocumentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root/'docs/platform').mkdir(parents=True)
        (self.root/'docs/platform/GUIDE.md').write_text('# Guide\n', encoding='utf8')
        (self.root/'README.md').write_text('[Guide](docs/platform/GUIDE.md)\n', encoding='utf8')
        (self.root/'.gitignore').write_text('docs/platform/validation/\n', encoding='utf8')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        subprocess.run(['git', 'add', 'README.md', '.gitignore', 'docs/platform/GUIDE.md'],
                       cwd=self.root, check=True, capture_output=True)

    def check_docs(self):
        output = StringIO()
        with patch.object(checker, 'ROOT', self.root), redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                checker.main()
        return raised.exception.code, output.getvalue()

    def test_ignored_experiment_documents_are_not_published_guides(self):
        directory = self.root/'docs/platform/validation'
        directory.mkdir()
        (directory/'report.md').write_text('[Missing](missing.md)\n', encoding='utf8')
        code, output = self.check_docs()
        self.assertEqual(code, False, output)
        self.assertIn('2 documents, 1 local links, 0 failures', output)

    def test_existing_but_untracked_link_target_is_rejected(self):
        directory = self.root/'docs/platform/validation'
        directory.mkdir()
        (directory/'report.md').write_text('# Local only\n', encoding='utf8')
        (self.root/'README.md').write_text('[Local](docs/platform/validation/report.md)\n', encoding='utf8')
        code, output = self.check_docs()
        self.assertEqual(code, True)
        self.assertIn('untracked', output)

    def add_html(self, source):
        path = self.root/'docs/platform/tutorial.html'
        path.write_text(source, encoding='utf8')
        subprocess.run(['git', 'add', 'docs/platform/tutorial.html'], cwd=self.root, check=True, capture_output=True)

    def test_html_navigation_and_duplicate_ids_are_checked(self):
        self.add_html('<a href="#missing">Read</a><section id="same"></section><section id="same"></section>')
        code, output = self.check_docs()
        self.assertTrue(code)
        self.assertIn('missing HTML anchor', output)
        self.assertIn('duplicate HTML id', output)

    def test_html_source_links_are_verified_against_the_checkout(self):
        self.add_html(f'<a href="{checker.REPOSITORY_BLOB}up/missing.py">Source</a>')
        code, output = self.check_docs()
        self.assertTrue(code)
        self.assertIn('up/missing.py', output)

    def test_self_contained_html_with_valid_links_passes(self):
        self.add_html('<nav><a href="#design">Design</a></nav><section id="design"></section>'
                      f'<a href="{checker.REPOSITORY_BLOB}docs/platform/GUIDE.md">Guide</a>')
        code, output = self.check_docs()
        self.assertFalse(code, output)


if __name__ == '__main__':
    unittest.main()
