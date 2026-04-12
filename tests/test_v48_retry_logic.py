import unittest
from unittest.mock import MagicMock, patch
from gaia_agent.models import _RetryWrapper
from gaia_agent.config import Config
from langchain_core.messages import HumanMessage

class TestRetryLogic(unittest.TestCase):
    def test_retry_on_resource_exhausted(self):
        # Mock the exception
        class MockResourceExhausted(Exception):
            pass
            
        # Manually inject our mock exception into RETRY_EXCEPTIONS for testing
        import gaia_agent.models
        original_exceptions = gaia_agent.models.RETRY_EXCEPTIONS
        gaia_agent.models.RETRY_EXCEPTIONS = (MockResourceExhausted,)
        
        # We need to re-initialize the retry logic with the new exception type
        # In a real scenario, this is done at module load time.
        # For testing, we'll mock the _retry_logic attribute
        from tenacity import retry, retry_if_exception_type, wait_none, stop_after_attempt
        
        mock_inner = MagicMock()
        # Fail twice with 429, then succeed
        mock_inner._generate.side_effect = [
            MockResourceExhausted("429 Quota Exceeded"),
            MockResourceExhausted("429 Quota Exceeded"),
            MagicMock(generations=[MagicMock(message=HumanMessage(content="Success"))])
        ]
        
        wrapper = _RetryWrapper(inner=mock_inner)
        # Override retry logic to be fast for tests
        wrapper._retry_logic = retry(
            retry=retry_if_exception_type(MockResourceExhausted),
            wait=wait_none(),
            stop=stop_after_attempt(5),
            reraise=True
        )
        
        print("Invoking wrapped model (expecting 2 retries)...")
        res = wrapper.invoke("hello")
        
        # Verify it was called 3 times
        self.assertEqual(mock_inner._generate.call_count, 3)
        self.assertEqual(res.content, "Success")
        print("✓ Retry logic successful: model retried after 429 errors and eventually succeeded.")

        # Clean up
        gaia_agent.models.RETRY_EXCEPTIONS = original_exceptions

if __name__ == "__main__":
    unittest.main()
