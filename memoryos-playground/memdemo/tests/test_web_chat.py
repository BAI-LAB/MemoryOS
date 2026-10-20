"""Flask -> MCP SDK -> response-generation tests, without GPU or model downloads.

Only the unchanged memory storage/embedding backends and LLM API are fixtures.
The chat route, Memoryos.get_response, MCP SDK and HTTP serialization are real.
Run: python -m unittest discover -s tests -v (from memdemo).
"""

import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from unittest.mock import Mock, patch

import httpx

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parent))
import web_context


def load_app():
    """Load production generation with isolated, unused heavy backend imports."""
    backends = {}
    exports = {
        'utils': ['OpenAIClient', 'get_timestamp', 'generate_id',
                  'gpt_user_profile_analysis', 'gpt_knowledge_extraction', 'ensure_directory_exists'],
        'short_term': ['ShortTermMemory'],
        'mid_term': ['MidTermMemory', 'compute_segment_heat'],
        'long_term': ['LongTermMemory'], 'updater': ['Updater'], 'retriever': ['Retriever'],
    }
    for name, names in exports.items():
        module = ModuleType(name)
        for export in names:
            setattr(module, export, Mock(return_value='2026-01-01 00:00:00'))
        backends[name] = module
    with patch.dict(sys.modules, backends):
        memory = importlib.import_module('memoryos')
        application = importlib.import_module('app')
    return application, memory.Memoryos


application, Memoryos = load_app()

REAL_HTTP_CLIENT = httpx.AsyncClient
RESULT = {'results': [{'title': 'Example source', 'url': 'https://example.org/news',
                       'excerpts': ['Current public facts.']} ]}


class MCPServer:
    """HTTP protocol fixture: exercises the actual MCP transport and SDK."""

    def __init__(self):
        self.requests = []
        self.error = False
        self.delay = 0
        self.status = 200
        self.payload = RESULT
        self.entered = threading.Event()
        self.lock = threading.Lock()

    async def handle(self, request):
        with self.lock:
            self.requests.append(request)
        if request.method != 'POST':
            return httpx.Response(405)
        body = json.loads(request.content)
        if 'id' not in body:
            return httpx.Response(202)
        if body['method'] == 'initialize':
            result = {'protocolVersion': body['params']['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'fixture', 'version': '1'}}
        elif body['method'] == 'tools/list':
            result = {'tools': [{'name': name, 'inputSchema': {'type': 'object'}}
                                for name in ['web_search', 'web_fetch']]}
        elif body['method'] == 'tools/call':
            self.entered.set()
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.status != 200:
                return httpx.Response(self.status, text='unavailable')
            result = {'content': [{'type': 'text', 'text': json.dumps(self.payload)}], 'isError': self.error}
        else:
            raise AssertionError(body)
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': result})

    def client(self, **kwargs):
        return REAL_HTTP_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)

    def calls(self):
        return [json.loads(r.content)['params'] for r in self.requests
                if r.method == 'POST' and json.loads(r.content).get('method') == 'tools/call']


class WebChatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.old_cwd = os.getcwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, self.old_cwd)
        self.server = MCPServer()
        self.http_patch = patch.object(web_context.httpx, 'AsyncClient', side_effect=self.server.client)
        self.http_patch.start()
        self.addCleanup(self.http_patch.stop)
        application.app.config.update(TESTING=True, WEB_SEARCH_ENABLED=True)
        application.memory_systems.clear()
        self.memory = Memoryos.__new__(Memoryos)
        self.memory.user_id = 'test-user'
        self.memory.assistant_id = 'test-assistant'
        self.memory.llm_model = 'saved-custom-model'
        self.memory.data_storage_path = self.temporary.name
        self.memory.user_data_dir = self.temporary.name + '/user'
        self.memory.assistant_data_dir = self.temporary.name + '/assistant'
        self.memory.retriever = Mock()
        self.memory.retriever.retrieve_context.return_value = {
            'retrieved_pages': [], 'retrieved_user_knowledge': [], 'retrieved_assistant_knowledge': []}
        self.memory.short_term_memory = Mock()
        self.memory.short_term_memory.get_all.return_value = [
            {'user_input': 'PRIVATE HISTORY', 'agent_response': 'PRIVATE ANSWER'}]
        self.memory.user_long_term_memory = Mock()
        self.memory.user_long_term_memory.get_raw_user_profile.return_value = 'PRIVATE PROFILE'
        self.memory.client = Mock()
        self.memory.client.chat_completion.return_value = 'Answer https://example.org/news'
        self.memory.add_memory = Mock()
        self.constructor = patch.object(application, 'Memoryos', return_value=self.memory)
        self.factory = self.constructor.start()
        self.addCleanup(self.constructor.stop)
        self.client = application.app.test_client()
        self.initialize(self.client)

    def initialize(self, client):
        response = client.post('/init_memory', json={'user_id': 'test-user', 'api_key': 'saved-llm-key',
                              'base_url': 'https://example.org/v1', 'model_name': 'saved-custom-model'})
        self.assertEqual(response.status_code, 200)

    def conversation_id(self, client):
        with client.session_transaction() as session:
            return session['web_conversation_id']

    def chat(self, message='Current public question', mode='search', client=None):
        return (client or self.client).post('/chat', json={'message': message, 'web_mode': mode})

    def test_anonymous_caller_consumes_sources_and_preserves_settings(self):
        with patch.dict(os.environ, {'PARALLEL_API_KEY': 'ambient-do-not-use', 'OPENAI_API_KEY': 'ambient'}):
            response = self.chat()
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json['sources'], [{'title': 'Example source', 'url': 'https://example.org/news'}])
        call = self.server.calls()[0]
        self.assertEqual(call['name'], 'web_search')
        self.assertEqual(call['arguments']['objective'], 'Current public question')
        self.assertEqual(call['arguments']['session_id'], self.conversation_id(self.client))
        for request in self.server.requests:
            self.assertEqual(str(request.url), web_context.MCP_URL)
            self.assertEqual(request.headers['User-Agent'], web_context.USER_AGENT)
            self.assertNotIn('Authorization', request.headers)
            self.assertNotIn('PRIVATE', request.content.decode())
            self.assertNotIn('saved-llm-key', request.content.decode())
        kwargs = self.memory.client.chat_completion.call_args.kwargs
        self.assertEqual(kwargs['model'], 'saved-custom-model')
        messages = json.dumps(kwargs['messages'])
        for text in ['PRIVATE PROFILE', 'PRIVATE HISTORY', 'Current public facts.', 'https://example.org/news', 'untrusted']:
            self.assertIn(text, messages)
        self.memory.add_memory.assert_called_once_with(user_input='Current public question',
            agent_response='Answer https://example.org/news', timestamp='2026-01-01 00:00:00')
        self.assertEqual(self.factory.call_args.kwargs['openai_api_key'], 'saved-llm-key')
        self.assertEqual(self.factory.call_args.kwargs['openai_base_url'], 'https://example.org/v1')

    def test_default_and_explicit_off_do_not_search(self):
        self.assertEqual(self.client.post('/chat', json={'message': 'hello'}).status_code, 200)
        self.assertEqual(self.chat('hello', 'off').status_code, 200)
        self.assertFalse(self.server.requests)
        self.assertEqual(self.client.post('/chat', json=[]).status_code, 400)
        self.assertEqual(self.client.post('/chat', data='bad json', content_type='application/json').status_code, 400)
        self.assertEqual(len(self.memory.client.chat_completion.call_args.kwargs['messages']), 2)

    def test_permission_and_invalid_inputs(self):
        application.app.config['WEB_SEARCH_ENABLED'] = False
        self.assertEqual(self.chat().status_code, 403)
        self.assertEqual(self.chat(mode='off').status_code, 200)
        application.app.config['WEB_SEARCH_ENABLED'] = True
        for message, mode in [('', 'search'), ('x' * 2001, 'search'), ('https://example.org', 'other'),
                              ('file:///etc/passwd', 'fetch'), ('https://u:p@example.org', 'fetch'),
                              ('https://example.org x', 'fetch'), (42, 'search')]:
            self.assertEqual(self.chat(message, mode).status_code, 400)
        self.assertFalse(self.server.requests)
        uninitialized = application.app.test_client()
        self.assertEqual(self.chat(client=uninitialized).status_code, 400)

    def test_search_fetch_followup_retry_and_reset_identity(self):
        first = self.conversation_id(self.client)
        self.assertEqual(len(first), 32)
        self.assertEqual(self.chat().status_code, 200)
        self.assertEqual(self.chat('https://example.org/news', 'fetch').status_code, 200)
        self.server.error = True
        self.server.payload = {'error': 'Free anonymous quota exceeded. Retry later.'}
        self.assertEqual(self.chat('follow-up').status_code, 502)
        self.server.error = False
        self.server.payload = RESULT
        self.assertEqual(self.chat('follow-up').status_code, 200)
        self.assertEqual({c['arguments']['session_id'] for c in self.server.calls()}, {first})
        self.assertEqual(self.server.calls()[1]['arguments']['urls'], ['https://example.org/news'])
        self.assertEqual(self.client.post('/clear_memory').status_code, 200)
        self.assertNotEqual(self.conversation_id(self.client), first)
        self.assertEqual(self.chat('after reset').status_code, 200)
        self.assertEqual(self.server.calls()[-1]['arguments']['session_id'], self.conversation_id(self.client))
        reset = self.conversation_id(self.client)
        self.initialize(self.client)
        self.assertNotEqual(self.conversation_id(self.client), reset)

    def test_interleaved_conversations_use_same_client_without_shared_identity(self):
        second = application.app.test_client()
        self.initialize(second)
        first_id, second_id = self.conversation_id(self.client), self.conversation_id(second)
        self.assertNotEqual(first_id, second_id)
        self.server.delay = 0.02
        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(self.chat, 'first conversation', 'search', self.client)
            b = executor.submit(self.chat, 'second conversation', 'search', second)
            self.assertEqual(a.result().status_code, 200)
            self.assertEqual(b.result().status_code, 200)
        observed = {c['arguments']['objective']: c['arguments']['session_id'] for c in self.server.calls()}
        self.assertEqual(observed, {'first conversation': first_id, 'second conversation': second_id})

    def test_errors_and_timeout_do_not_generate_or_save_fallback(self):
        for condition in ['tool_error', 'http_error', 'invalid_payload', 'page_error', 'timeout']:
            with self.subTest(condition=condition):
                self.server.error = condition == 'tool_error'
                self.server.status = 503 if condition == 'http_error' else 200
                self.server.payload = {'oops': []} if condition == 'invalid_payload' else (
                    {'results': [], 'errors': ['page unavailable']} if condition == 'page_error' else RESULT)
                self.server.delay = 1 if condition == 'timeout' else 0
                with patch.object(web_context, 'TIMEOUT_SECONDS', 0.05):
                    self.assertEqual(self.chat().status_code, 502)
        self.memory.client.chat_completion.assert_not_called()
        self.memory.add_memory.assert_not_called()

    def test_cancellation_closes_transport_and_propagates(self):
        self.server.delay = 60

        async def cancel():
            task = asyncio.create_task(application.web_client.retrieve('cancel me', 'search',
                                        self.conversation_id(self.client)))
            while not self.server.entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(cancel())
        self.memory.client.chat_completion.assert_not_called()

    def test_page_and_source_rendering_are_safe(self):
        page = self.client.get('/').text
        self.assertIn('value="off" selected', page)
        self.assertIn("document.getElementById('webMode').value = 'off'", page)
        self.assertIn('messageDiv.textContent = message', page)
        self.assertIn('link.textContent = source.title || source.url', page)
        self.assertIn("['http:', 'https:'].includes(url.protocol)", page)
        self.assertIn("link.rel = 'noopener noreferrer'", page)
        self.server.payload = {'results': [{'url': 'javascript:alert(1)', 'excerpts': ['bad']},
            {'url': 'https://example.org', 'title': '<img src=x onerror=alert(1)>', 'excerpts': ['safe data']}]}
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['sources']), 1)
        self.assertEqual(response.json['sources'][0]['url'], 'https://example.org')


if __name__ == '__main__':
    unittest.main()
