"""Comprehensive unit tests for the LiteLLMClient class.

These tests mock litellm.completion and litellm exception classes so
that litellm does NOT need to be installed in the test environment.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Bootstrap a fake ``litellm`` package so ``LiteLLMClient`` can be imported
# without the real library installed.
# ---------------------------------------------------------------------------
def _build_fake_litellm():
    """Create minimal stub modules for litellm + litellm.exceptions."""

    fake_litellm = types.ModuleType("litellm")
    fake_exceptions = types.ModuleType("litellm.exceptions")

    # Base exception all litellm errors derive from
    class _LiteLLMError(Exception):
        def __init__(self, message="", model=None, llm_provider=None, response=None):
            self.message = message
            self.model = model
            self.llm_provider = llm_provider
            self.response = response
            super().__init__(message)

    class AuthenticationError(_LiteLLMError):
        pass

    class RateLimitError(_LiteLLMError):
        pass

    class APIConnectionError(_LiteLLMError):
        pass

    class Timeout(_LiteLLMError):
        pass

    class NotFoundError(_LiteLLMError):
        pass

    class BadRequestError(_LiteLLMError):
        pass

    class ContextWindowExceededError(_LiteLLMError):
        pass

    # Attach to the exceptions sub-module
    for cls in (
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        Timeout,
        NotFoundError,
        BadRequestError,
        ContextWindowExceededError,
    ):
        setattr(fake_exceptions, cls.__name__, cls)
        setattr(fake_litellm, cls.__name__, cls)

    fake_litellm.exceptions = fake_exceptions
    fake_litellm.completion = MagicMock()  # default; overridden per test

    return fake_litellm, fake_exceptions


_fake_litellm, _fake_exceptions = _build_fake_litellm()

# Inject the fake modules before importing the code under test
sys.modules["litellm"] = _fake_litellm
sys.modules["litellm.exceptions"] = _fake_exceptions

# Add the memoryos-pypi directory to sys.path so ``utils`` can be
# imported directly.  Remove the parent package's __init__.py from
# the import chain to avoid triggering ``from .memoryos import ...``
# which requires a full package context.
_PACKAGE_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
_PACKAGE_DIR = os.path.abspath(_PACKAGE_DIR)
sys.path.insert(0, _PACKAGE_DIR)

# Prevent pytest from importing memoryos-pypi/__init__.py which does
# a relative import that fails outside of a proper install.
sys.modules.setdefault("memoryos", types.ModuleType("memoryos"))

from utils import LiteLLMClient  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_response(content="Hello"):
    """Build a mock object shaped like a litellm completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


SAMPLE_MESSAGES = [{"role": "user", "content": "Hi"}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLiteLLMClientHappyPath(unittest.TestCase):
    """1. Happy path - valid response."""

    @patch("litellm.completion")
    def test_returns_content(self, mock_completion):
        mock_completion.return_value = _make_response("Hello")
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertEqual(result, "Hello")


class TestLiteLLMClientAuthError(unittest.TestCase):
    """2. Invalid / expired API key."""

    @patch("litellm.completion")
    def test_authentication_error(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.AuthenticationError(
            "Invalid API key"
        )
        client = LiteLLMClient(api_key="bad-key", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("Authentication", result)


class TestLiteLLMClientNotFound(unittest.TestCase):
    """3. Model not found."""

    @patch("litellm.completion")
    def test_not_found_error(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.NotFoundError(
            "Model not found"
        )
        client = LiteLLMClient(api_key="sk-test", model="nonexistent-model")
        result = client.chat_completion(
            model="nonexistent-model", messages=SAMPLE_MESSAGES
        )
        self.assertIn("not found", result.lower())


class TestLiteLLMClientRateLimit(unittest.TestCase):
    """4. Rate limit (429)."""

    @patch("litellm.completion")
    def test_rate_limit_error(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.RateLimitError(
            "Rate limit exceeded"
        )
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("Rate limit", result)


class TestLiteLLMClientContextWindow(unittest.TestCase):
    """5. Token limit exceeded."""

    @patch("litellm.completion")
    def test_context_window_exceeded(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.ContextWindowExceededError(
            "Context window exceeded"
        )
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("context window", result.lower())


class TestLiteLLMClientEmptyResponse(unittest.TestCase):
    """6. Empty / null response."""

    @patch("litellm.completion")
    def test_none_content_returns_error(self, mock_completion):
        mock_completion.return_value = _make_response(content=None)
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("empty response", result.lower())


class TestLiteLLMClientTimeout(unittest.TestCase):
    """7. Timeout."""

    @patch("litellm.completion")
    def test_timeout_error(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.Timeout("Request timed out")
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("timed out", result.lower())


class TestLiteLLMClientConnectionError(unittest.TestCase):
    """8. Connection error."""

    @patch("litellm.completion")
    def test_connection_error(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.APIConnectionError(
            "Connection refused"
        )
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("connect", result.lower())


class TestLiteLLMClientBadRequest(unittest.TestCase):
    """9. Bad request."""

    @patch("litellm.completion")
    def test_bad_request_error(self, mock_completion):
        mock_completion.side_effect = _fake_exceptions.BadRequestError(
            "Invalid request"
        )
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        result = client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        self.assertIn("Bad request", result)


class TestLiteLLMClientDropParams(unittest.TestCase):
    """10. drop_params=True is passed."""

    @patch("litellm.completion")
    def test_drop_params_true(self, mock_completion):
        mock_completion.return_value = _make_response("ok")
        client = LiteLLMClient(api_key="sk-test", model="gpt-4o-mini")
        client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        call_kwargs = mock_completion.call_args[1]
        self.assertTrue(call_kwargs.get("drop_params"), "drop_params should be True")


class TestLiteLLMClientAPIKeyForwarding(unittest.TestCase):
    """11. API key forwarding."""

    @patch("litellm.completion")
    def test_api_key_forwarded(self, mock_completion):
        mock_completion.return_value = _make_response("ok")
        client = LiteLLMClient(api_key="test-key", model="gpt-4o-mini")
        client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs.get("api_key"), "test-key")


class TestLiteLLMClientAPIKeyOmitted(unittest.TestCase):
    """12. API key omitted when None (env-var fallback)."""

    @patch("litellm.completion")
    def test_api_key_not_in_kwargs_when_none(self, mock_completion):
        mock_completion.return_value = _make_response("ok")
        client = LiteLLMClient(api_key=None, model="gpt-4o-mini")
        client.chat_completion(model="gpt-4o-mini", messages=SAMPLE_MESSAGES)
        call_kwargs = mock_completion.call_args[1]
        self.assertNotIn(
            "api_key",
            call_kwargs,
            "api_key should NOT be in call kwargs when client api_key is None",
        )


if __name__ == "__main__":
    unittest.main()
