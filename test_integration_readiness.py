"""Non-model regression checks for deployable research-demo packaging."""
from html.parser import HTMLParser
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class Elements(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.elements = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class IntegrationReadinessTests(unittest.TestCase):
    def test_every_demo_has_same_origin_csp_and_no_inline_scripts(self):
        for name in ['index.html', 'headlines.html', 'predict.html']:
            with self.subTest(name=name):
                parsed = Elements((ROOT / 'docs' / name).read_text())
                policies = [a['content'] for t, a in parsed.elements if t == 'meta' and a.get('http-equiv', '').lower() == 'content-security-policy']
                self.assertEqual(len(policies), 1)
                for directive in ["default-src 'self'", "script-src 'self'", "object-src 'none'", "base-uri 'none'"]:
                    self.assertIn(directive, policies[0])
                self.assertNotIn('unsafe-inline', policies[0])
                self.assertTrue(all(a.get('src') for t, a in parsed.elements if t == 'script'))

    def test_archived_pages_point_to_current_input_demo(self):
        for name in ['index.html', 'headlines.html']:
            content = (ROOT / 'docs' / name).read_text()
            self.assertIn('Archived', content)
            self.assertTrue(any(t == 'a' and a.get('href') == 'predict.html' for t, a in Elements(content).elements))

    def test_tables_are_named_keyboard_focusable_scroll_regions(self):
        content = (ROOT / 'docs/headlines.html').read_text()
        regions = [a for t, a in Elements(content).elements if 'scroll' in a.get('class', '').split()]
        self.assertEqual(len(regions), 2)
        for region in regions:
            self.assertEqual(region.get('tabindex'), '0')
            self.assertEqual(region.get('role'), 'region')
            self.assertTrue(region.get('aria-label'))
        self.assertIn('Scroll horizontally', content)

    def test_supported_runtime_is_declared(self):
        self.assertEqual((ROOT / '.python-version').read_text().strip(), '3.12')
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('Python 3.12', readme)
        self.assertIn('python verify_submission.py', readme)


if __name__ == '__main__':
    unittest.main()
