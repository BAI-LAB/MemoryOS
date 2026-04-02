"""
PoC test for CWE-22 path traversal via user_id in Flask web app.

Tests that the /init_memory endpoint rejects user_id values containing
path traversal sequences (e.g., '..', '/', '\0').
"""
import sys
import os
import json
import types

# We test the Flask app directly via its test client - no need for the full
# Memoryos stack (which requires ML models). We only need to verify that
# the input validation rejects malicious user_id values before they reach
# the Memoryos constructor.

WORKTREE = os.environ.get('WORKTREE', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock out the heavy memoryos imports before loading the Flask app
class MockMemoryos:
    """Captures constructor args so we can inspect what user_id was passed."""
    instances = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        MockMemoryos.instances.append(self)
        self.user_data_dir = os.path.join(
            kwargs.get('data_storage_path', '/tmp'),
            'users',
            kwargs.get('user_id', '')
        )
        self.assistant_data_dir = os.path.join(
            kwargs.get('data_storage_path', '/tmp'),
            'assistants',
            kwargs.get('assistant_id', '')
        )


mock_memoryos_module = types.ModuleType('memoryos')
mock_memoryos_module.Memoryos = MockMemoryos

mock_utils_module = types.ModuleType('memoryos.utils')
mock_utils_module.get_timestamp = lambda: "2025-01-01 00:00:00"

sys.modules['memoryos'] = mock_memoryos_module
sys.modules['memoryos.utils'] = mock_utils_module

# Now add the app directory and import it
sys.path.insert(0, os.path.join(WORKTREE, 'memoryos-playground', 'memdemo'))
import app as flask_app

client = flask_app.app.test_client()


def post_init(user_id):
    """Helper: POST /init_memory with the given user_id."""
    return client.post('/init_memory', json={
        'user_id': user_id,
        'api_key': 'sk-test-fake',
        'base_url': 'http://localhost:9999',
        'model_name': 'gpt-4o-mini',
    })


# ---- Attack vectors ----

TRAVERSAL_PAYLOADS = [
    '../etc/cron.d',               # classic unix traversal
    '..\\windows\\system32',       # windows traversal
    'foo/../../etc/passwd',        # embedded traversal
    'foo/../../../tmp/evil',       # deeper traversal
    '....//....//etc',             # double-dot-slash bypass
    '/absolute/path',             # absolute path
    '.',                           # current directory reference
    '..',                          # parent directory reference
]

SAFE_PAYLOADS = [
    'alice',
    'bob_123',
    'user-2024',
    'CamelCaseUser',
]


def test_traversal_payloads_rejected():
    """All path-traversal payloads must be rejected with 400."""
    for payload in TRAVERSAL_PAYLOADS:
        resp = post_init(payload)
        assert resp.status_code == 400, (
            f"VULN: user_id={payload!r} was accepted (status {resp.status_code}). "
            f"Response: {resp.get_data(as_text=True)}"
        )
        body = resp.get_json()
        assert 'error' in body, f"Expected error key in response for {payload!r}"
        print(f"  PASS rejected: {payload!r} -> 400")


def test_safe_payloads_accepted():
    """Legitimate user_id values must still be accepted."""
    for payload in SAFE_PAYLOADS:
        MockMemoryos.instances.clear()
        resp = post_init(payload)
        body = resp.get_json()
        if resp.status_code == 400:
            err = body.get('error', '').lower()
            assert not ('user_id' in err and 'invalid' in err), (
                f"False positive: safe user_id={payload!r} was rejected: {body}"
            )
        print(f"  PASS accepted: {payload!r} -> {resp.status_code}")


if __name__ == '__main__':
    print("--- Testing path traversal payloads are rejected ---")
    try:
        test_traversal_payloads_rejected()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

    print("\n--- Testing safe payloads are accepted ---")
    try:
        test_safe_payloads_accepted()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

    print("\nAll tests passed!")
