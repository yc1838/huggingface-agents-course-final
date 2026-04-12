import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found, using existing environment variables.")

def test_gemma_instructions():
    api_key = os.getenv("GAIA_GOOGLE_API_KEY")
    if not api_key:
        print("Error: GAIA_GOOGLE_API_KEY not found in .env")
        return

    # Initialize Gemma 4 Instruct (The 31B Dense model is great for reasoning)
    print("Initializing gemma-4-31b-it...")
    model = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=api_key,
        temperature=0.0
    )

    # Test Case 1: Complex Formatting / JSON Extraction
    print("\n--- Test 1: JSON Extraction ---")
    prompt = (
        "Extract the following into a JSON object with keys 'author', 'title', and 'year'. "
        "Text: 'In 1994, Yujing Chen published a fascinating paper titled The Agent Architectures of Tomorrow.'"
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        print(f"Response:\n{response.content}")
    except Exception as e:
        print(f"Test 1 Failed: {e}")

    # Test Case 2: Multi-step Reasoning / Instruction Following
    print("\n--- Test 2: Reasoning & Planning ---")
    system_instr = "You are a logical solver. Do not yap. Only provide the final result."
    reasoning_prompt = (
        "If a train leaves Station A at 10:00 AM moving at 50 mph, "
        "and another train leaves Station B (100 miles away) at 10:30 AM moving at 100 mph toward Station A, "
        "at what time (HH:MM AM/PM) will they meet?"
    )
    try:
        response = model.invoke([
            SystemMessage(content=system_instr),
            HumanMessage(content=reasoning_prompt)
        ])
        print(f"Response:\n{response.content}")
    except Exception as e:
        print(f"Test 2 Failed: {e}")

if __name__ == "__main__":
    test_gemma_instructions()
