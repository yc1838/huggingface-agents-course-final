import unittest
from unittest.mock import MagicMock
from gaia_agent.models import _RetryWrapper, _RETRY_PARAMS
from langchain_core.messages import HumanMessage
from tenacity import retry_if_exception_type

class TestRetryFix(unittest.TestCase):
    def test_retry_arguments_passing(self):
        """Verify that _RetryWrapper correctly passes arguments to the inner model."""
        
        # 1. Create a dummy exception type
        class Mock429(Exception):
            pass
            
        # 2. Inject it into our retry params for this test session
        import gaia_agent.models
        original_exceptions = gaia_agent.models.RETRY_EXCEPTIONS
        gaia_agent.models.RETRY_EXCEPTIONS = (Mock429,)
        
        # Update the global params to recognize our mock exception
        gaia_agent.models._RETRY_PARAMS['retry'] = retry_if_exception_type(Mock429)
        # Disable wait for speed
        original_wait = gaia_agent.models._RETRY_PARAMS['wait']
        from tenacity import wait_none
        gaia_agent.models._RETRY_PARAMS['wait'] = wait_none()

        mock_inner = MagicMock()
        # Side effect: fail once, succeed once
        mock_inner._generate.side_effect = [
            Mock429("Rate limit hit"),
            MagicMock(generations=[MagicMock(message=HumanMessage(content="Fixed!"))])
        ]
        
        wrapper = _RetryWrapper(inner=mock_inner)
        
        print("Invoking model with positional arguments...")
        # invoke calls _generate internally
        try:
            res = wrapper.invoke("Test prompt")
            print(f"Result: {res.content}")
            
            # 3. Assertions
            self.assertEqual(res.content, "Fixed!")
            self.assertEqual(mock_inner._generate.call_count, 2)
            print("✓ Argument passing verified: No positional argument errors!")
        except TypeError as e:
            self.fail(f"TypeError caught: {e}. Fix failed.")
        finally:
            # Clean up
            gaia_agent.models.RETRY_EXCEPTIONS = original_exceptions
            gaia_agent.models._RETRY_PARAMS['retry'] = retry_if_exception_type(original_exceptions) if original_exceptions else lambda e: False
            gaia_agent.models._RETRY_PARAMS['wait'] = original_wait

if __name__ == "__main__":
    unittest.main()
