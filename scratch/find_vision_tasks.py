from gaia_agent.gaia_dataset import GaiaDatasetClient
import os

def find_vision_tasks():
    token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")
    # Using L2 validation split as that's what we've been running
    client = GaiaDatasetClient(config="2023_all", split="validation", level="2", token=token)
    questions = client.get_questions()
    
    vision_keywords = ["image", "photo", "picture", "screenshot", "diagram", "figure", "visual", "jpg", "png", "jpeg"]
    
    matches = []
    for q in questions:
        q_text = q['question'].lower()
        if any(kw in q_text for kw in vision_keywords):
            matches.append({
                "task_id": q['task_id'],
                "question": q['question'][:200] + "..."
            })
            
    print(f"Found {len(matches)} potential vision tasks in Level 2:")
    for m in matches:
        print(f"- ID: {m['task_id']}\n  Q: {m['question']}\n")

if __name__ == "__main__":
    find_vision_tasks()
