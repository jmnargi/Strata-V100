"""Bounded, opt-in continuation after a native-tool action preamble stops without a call."""
import json
import threading
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, _ends_in_tool_preamble, serve

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = ("Still over-long, and the inline dict-comprehension monsters keep producing invalid syntax. "
            "Decisive fix: **the long rationale belongs in FINDINGS.md, not in the tool** — short docstring, "
            "plain loops, build dicts with simple statements. Writing minimal and validating immediately:")
TOOLS = [{"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
ARGUMENTS = '<parameter=command>\nprintf health-check\n</parameter>\n</function>\n</tool_call>'
CALL = '<tool_call>\n<function=bash>\n' + ARGUMENTS


class RecordingEngine(MockEngine):
    def __init__(self, tok, scripts):
        super().__init__(tok, scripts, max_context=100000)
        self.prompts = []
        self.caps = []

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.prompts.append(list(ids))
        self.caps.append(max_new)
        yield from super().generate(ids, max_new, sampling, cancel, **kwargs)


class ToolCallContinuation(unittest.TestCase):
    def run_reply(self, text=PREAMBLE, enabled=True, tools=TOOLS, cap=2048, sampling=None, cancelled=False):
        tok = ByteTokenizer()
        engine = RecordingEngine(tok, [text, ARGUMENTS, ARGUMENTS])
        svc = Service(engine, tok, None)
        svc.tool_call_continuation = enabled
        cancel = threading.Event()
        if cancelled:
            cancel.set()
        events = list(svc.run(tok.encode('Health check'), False, tools, cap, sampling or {}, cancel))
        return svc, engine, events

    def test_reported_preamble_continues_once_with_shared_output_cap(self):
        svc, engine, events = self.run_reply()
        calls = [e.call for kind, e in events if kind == 'event' and e.kind == 'tool_call']
        self.assertEqual([(c.name, c.arguments) for c in calls], [('bash', {'command': 'printf health-check'})])
        self.assertEqual(len(engine.prompts), 2)
        self.assertEqual(svc.totals['tool_call_continuations'], 1)
        done = events[-1][1]
        self.assertLessEqual(done['completion_tokens'], 2048)
        self.assertLessEqual(engine.caps[1], 2048 - (len(engine.prompts[1]) - len(engine.prompts[0])))

    def test_default_path_and_request_boundaries_do_not_continue(self):
        for options in ({'enabled': False}, {'tools': []}, {'cap': 20},
                        {'cap': len(PREAMBLE.encode()) + 30}, {'cancelled': True},
                        {'sampling': {'stop': ['immediately:']}}, {'sampling': {'stop': ['not present']}}):
            with self.subTest(options=options):
                svc, engine, events = self.run_reply(**options)
                self.assertLessEqual(len(engine.prompts), 1)
                self.assertEqual(svc.totals.get('tool_call_continuations', 0), 0)
                self.assertFalse(any(kind == 'event' and e.kind == 'tool_call' for kind, e in events))

    def test_answers_quotes_and_examples_do_not_continue(self):
        for text in ('Done. All checks passed.', 'The results are:', 'Writing is useful.',
                     'I will not run this:', 'I could run this:', 'Let me explain the result:',
                     '```text\nWriting minimal and validating immediately:\n```',
                     '> Writing minimal and validating immediately:',
                     '"Writing minimal and validating immediately:"',
                     'Example:\nWriting minimal and validating immediately:'):
            with self.subTest(text=text):
                self.assertFalse(_ends_in_tool_preamble(text))
                svc, engine, events = self.run_reply(text=text)
                self.assertEqual(len(engine.prompts), 1)

    def test_existing_or_unfinished_calls_do_not_trigger_another_call(self):
        for text in (CALL + '\n' + PREAMBLE, CALL[:-20] + PREAMBLE):
            with self.subTest(text=text):
                svc, engine, events = self.run_reply(text=text)
                self.assertEqual(len(engine.prompts), 1)
                self.assertEqual(svc.totals.get('tool_call_continuations', 0), 0)

    def test_no_unbounded_retry_when_continuation_fails(self):
        tok = ByteTokenizer()
        engine = RecordingEngine(tok, [PREAMBLE, '', ARGUMENTS])
        svc = Service(engine, tok, None)
        svc.tool_call_continuation = True
        events = list(svc.run(tok.encode('Check'), False, TOOLS, 2048, {}, threading.Event()))
        self.assertEqual(len(engine.prompts), 2)
        self.assertFalse(any(kind == 'event' and e.kind in ('tool_start', 'tool_call') for kind, e in events))

    def test_streamed_and_whole_http_return_model_selected_tool(self):
        for stream in (False, True):
            for choice in ('auto', 'none'):
                with self.subTest(stream=stream, choice=choice):
                    tok = ByteTokenizer()
                    engine = RecordingEngine(tok, ['Thought.</think>\n' + PREAMBLE, 'bash>\n' + ARGUMENTS])
                    svc = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
                    svc.tool_call_continuation = True
                    httpd = serve(svc, port=0)
                    try:
                        body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'Health check'}],
                                'tools': [{'type': 'function', 'function': t} for t in TOOLS + [
                                    {'name': 'read', 'parameters': {'type': 'object', 'properties': {}}}]],
                                'tool_choice': choice, 'max_tokens': 2048, 'stream': stream}
                        req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req, timeout=20) as response:
                            raw = response.read().decode()
                        if stream:
                            chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                            choices = [c for chunk in chunks for c in chunk.get('choices', [])]
                            calls = [t for c in choices for t in c.get('delta', {}).get('tool_calls', [])]
                            names = [t['function']['name'] for t in calls if t.get('function', {}).get('name')]
                            args = ''.join(t.get('function', {}).get('arguments', '') for t in calls)
                            finish = [c['finish_reason'] for c in choices if c.get('finish_reason')][-1]
                        else:
                            result = json.loads(raw)['choices'][0]
                            calls = result['message'].get('tool_calls', [])
                            names = [t['function']['name'] for t in calls]
                            args = calls[0]['function']['arguments'] if calls else ''
                            finish = result['finish_reason']
                        self.assertEqual(names, ['bash'] if choice == 'auto' else [])
                        self.assertEqual(finish, 'tool_calls' if choice == 'auto' else 'stop')
                        if choice == 'auto':
                            self.assertEqual(json.loads(args), {'command': 'printf health-check'})
                    finally:
                        httpd.shutdown()
                        httpd.server_close()
