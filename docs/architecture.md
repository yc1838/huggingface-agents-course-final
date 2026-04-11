# GAIA Agent Enhanced Architecture

This diagram illustrates the current refined architecture of our GAIA agent, incorporating the **Planner Recovery Mode** and the **Task Chronicle** system.

```mermaid
graph TD
    %% Entry Point
    Start((Start)) --> Planner[Planner Node]

    %% Main Reasoning Loop
    subgraph "Reasoning & Execution Loop"
        Planner --> Orchestrator{Orchestrator}
        Orchestrator -- "Needs Tool Step" --> Executor[Executor Node]
        Executor -- "Tool Results" --> Reflector[Reflector Node]
        Reflector -- "Memory Upgrade & Chronicle Update" --> Orchestrator
    end

    %% State & Memory Management
    subgraph "State Management (AgentState)"
        Chronicle[(Task Chronicle\nPersistent Facts)]
        WorkingMemory[Working Memory\nRaw Context]
    end

    %% Verification & Recovery
    Orchestrator -- "Draft Answer Ready" --> Verifier{Verifier Node}
    Verifier -- "APPROVED" --> End((Success))
    Verifier -- "REJECTED (Critique)" --> Recovery{Recovery Mode?}
    
    Recovery -- "Patch Existing Plan" --> Planner
    Recovery -- "Full Restart" --> Planner

    %% Data Flow Connections
    Reflector -.-> Chronicle
    Chronicle -.-> Planner
    Chronicle -.-> Orchestrator
    
    %% Cache Layer
    LLMCache{{LLM Cache\nSQLite DB}} -.-> Planner
    LLMCache -.-> Orchestrator
    LLMCache -.-> Reflector
    LLMCache -.-> Verifier
```

### Key Components:

1.  **Task Chronicle**: A central, persistent memory that stores only definitive facts (e.g., "USDA 1959 document found"). It survives even when a plan is rejected.
2.  **Planner Recovery Mode**: When the Verifier gives a critique (e.g., "wrong format"), the Planner uses the Chronicle and the critique to generate a "patch" step instead of restarting the whole research process.
3.  **Orchestrator**: Acts as the executive brain, checking the Chronicle first to see if a final answer can be synthesized early.
4.  **Reflector**: Crucial for state maintenance; it extracts `CHRONICLE UPDATE` lines from tool results to keep the mission on track.
5.  **LLM Cache**: Speeds up the entire loop by skipping identical LLM triggers for repeated runs or debugging.
