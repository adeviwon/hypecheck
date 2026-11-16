"""Guard the judge-facing reading path without changing the scientific evaluation."""
import hashlib
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent


class ReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads((ROOT / 'HypeCheck.ipynb').read_text())

    def test_no_embedded_source_or_data_walls(self):
        # Infrastructure cells stay tiny. The one cell a judge is meant to edit holds the
        # 40 demo posts, so it is bounded separately and must stay exactly 40 posts.
        demo = 0
        for cell in self.nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            self.assertNotIn('EMBEDDED_INPUTS =', source)
            self.assertNotIn('SOURCES =', source)
            if 'TWEETS = [' in source:
                demo += 1
                self.assertLessEqual(len(source), 8000)
                self.assertEqual(sum(1 for line in source.splitlines() if line.startswith('    ')), 40)
                continue
            self.assertLessEqual(len(source), 2400)
            self.assertLessEqual(len(source.splitlines()), 24)
        self.assertEqual(demo, 1, 'the editable tweet block must exist exactly once')

    def test_reading_route_and_plain_language(self):
        text = '\n'.join(''.join(cell['source']) for cell in self.nb['cells'] if cell['cell_type'] == 'markdown')
        for phrase in ('Who this is for:', 'Bottom line:', 'Runtime → Run all', '## 1. Load',
                       '## 4. Baselines', '## 6. Where', '## 8. Improvement',
                       '## 9. The demo', '## 10. Results', '## 11. Limitations',
                       'Brier error', 'not independent confirmation', 'not fine-tuned'):
            self.assertIn(phrase, text)
        self.assertTrue(self.nb['metadata']['colab']['toc_visible'])

    def test_reference_training_and_error_analysis_are_executed(self):
        code="\n".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"]=="code")
        for fragment in ["model1.fit(X_train", "model2.fit(X_train", "class_weight=\"balanced\"", "ngram_range=(2,5)", "C=10", "def inspect_errors", "actual_word_reasons", "error_type", "Missed UP", "False UP", "run_experiment(trainer=use_visible_models)"]:
            self.assertIn(fragment,code)

    def test_no_reference_comparisons_outside_source_credits(self):
        for cell in self.nb["cells"]:
            if cell["cell_type"]=="markdown":
                text="".join(cell["source"])
                if not text.startswith("## Sources"):
                    self.assertNotIn("ScamCheck",text)
                    self.assertNotIn("Like the sample",text)
    def test_examples_and_findings_are_human_readable(self):
        code="\n".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"]=="code")
        text="\n".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"]=="markdown")
        for phrase in ["examples_table(train)","scores_table(","results_table(report)","What these examples show","What this result means"]:self.assertIn(phrase,code)
        self.assertIn("Our findings, in plain English",text)
        self.assertIn("not the probability",text.lower())

    def test_judges_get_current_input_blocks_without_editing_code(self):
        code = [''.join(cell['source']) for cell in self.nb['cells'] if cell['cell_type'] == 'code']
        source = '\n'.join(code)
        # a one-tweet form that runs before training, and the editable 40-post block
        self.assertIn('#@title Check a custom tweet', source)
        self.assertIn('#@title Enter tweets here', source)
        self.assertIn('TWEETS = [', source)
        self.assertIn('td.run_block(TWEETS,COIN,artifact,known=td.examples())', source)
        # the removed interactive app must not come back
        self.assertNotIn('RUN_APP', source)
        self.assertNotIn('build_app(', source)
        for name in ['index.html', 'headlines.html', 'predict.html']:
            page = (ROOT / 'docs' / name).read_text()
            self.assertIn('https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb', page)

    def test_verified_source_and_support_parity(self):
        import colab_setup
        metadata = self.nb['metadata']['hypecheck']
        for name, digest in metadata['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
            config=json.loads((ROOT / "crowd-source.json").read_text())
            self.assertEqual(config["files"][name], digest)
        for name, digest in metadata['bootstrap_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_custom_tweet_form_runs_before_training(self):
        import pandas as pd

        code = [''.join(cell['source']) for cell in self.nb['cells'] if cell['cell_type'] == 'code']
        form = code[2]
        self.assertTrue(form.startswith('#@title Check a custom tweet'))
        pin = json.loads((ROOT / 'crowd-source.json').read_text())['trained_model']
        payload = (ROOT / pin['path']).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), pin['sha256'])
        fetch = Mock(return_value=payload)
        score = Mock()
        namespace = {'pd': pd, 'json': json, 'verified_bytes': fetch, 'batch_check': score,
                     'display': Mock(), 'HTML': str}
        with patch('builtins.print'):
            exec(form, namespace)
        fetch.assert_not_called()
        score.assert_not_called()

        with patch('crowd_app.panels', return_value=('sentiment', 'forecast', 'evidence')):
            for tweet in ['DOGE discussion\n\ncontinuation', 'Another tweet about DOGE']:
                exec(form.replace('tweet_text = ""', f'tweet_text = {tweet!r}', 1), namespace)
                text, coin = score.call_args.args
                self.assertEqual(pd.read_csv(StringIO(text))['text'].tolist(), [tweet])
                self.assertEqual(coin, 'DOGE')
                self.assertEqual(score.call_args.kwargs['input_format'], 'csv')
        fetch.assert_called_once_with(pin['path'], pin['revision'], pin['sha256'])
        self.assertEqual(namespace['display'].call_count, 6)

    def test_rich_table_escapes_source_text(self):
        import pandas as pd
        from notebook_support import table
        with patch('IPython.display.display') as display:
            table(pd.DataFrame({'Post': ['<script>not trusted</script>']}), '<caption>')
        output = display.call_args.args[0].data
        self.assertIn('&lt;script&gt;', output)
        self.assertNotIn('<script>', output)
        self.assertIn('&lt;caption&gt;', output)

    def test_chart_uses_actual_scores_and_full_scale(self):
        from notebook_support import accuracy_chart
        report = json.loads((ROOT / 'docs/headline-report.json').read_text())
        with patch('IPython.display.display') as display:
            accuracy_chart(report, ['Always up', 'Sentiment + coin'])
        output = display.call_args.args[0].data
        self.assertIn(f"{report['metrics']['Sentiment + coin']['accuracy']:.2%}", output)
        self.assertIn('100%', output)
        self.assertIn('0%', output)
        self.assertIn('aria-label', output)

    def test_saved_outputs_include_tables_and_chart(self):
        outputs = [output for cell in self.nb['cells'] if cell['cell_type'] == 'code'
                   for output in cell.get('outputs', [])]
        tables = [output for output in outputs if 'text/html' in output.get('data', {})]
        charts = [output for output in outputs if 'image/svg+xml' in output.get('data', {})]
        self.assertGreaterEqual(len(tables), 8)
        self.assertEqual(len(charts), 1)
        self.assertFalse(any(output['output_type'] == 'error' for output in outputs))


if __name__ == '__main__':
    unittest.main()
