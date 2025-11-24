# Team Agent Pattern

**Author**: Wei Sun
**Status**: Draft
**Created**: 2025-11

## Problem

### Current State: Tree Structure with LLM-Based Transfer

ADK's official library uses a hierarchical tree structure with LLM-based agent transfer. Agents can transfer to child nodes and peers within the tree, but the structure has fundamental limitations:

- **Static tree hierarchy** - Scope of collaboration limited by tree structure (children, peers)
- **Cannot adapt to dynamic problem scopes** - Experts may need to collaborate in different combinations based on context
- **Limited by predefined relationships** - Can only transfer within statically defined tree relationships
- **LLM routing doesn't solve inflexibility** - Even with intelligent routing, the static tree structure constrains collaboration patterns

### Why We Need Team Agents

Cooperative multi-agent patterns require:

- **Flat team structure** - Experts can collaborate in any combination, not limited by tree hierarchy
- **Dynamic collaboration patterns** - Adapt expert combinations based on actual problem scope
- **Explicit coordination primitives** - Delegate, handoff, escalate as first-class operations
- **Configurable peer relationships** - `allowed_peers` enables flexible collaboration without rigid tree constraints

**Example**: Customer support needs order_specialist + billing_specialist for billing issues, but order_specialist + shipping_specialist for delivery issues. Tree structures with static relationships cannot adapt these collaboration patterns dynamically based on problem context.

## Motivation

### User Benefits

- **Intuitive team structure** - Mirrors real-world teams: one coordinator + multiple specialists
- **Clear coordination patterns** - Standardized delegation, handoff, and escalation primitives
- **Specialist reusability** - Same specialist can work in multiple teams
- **Simplified implementation** - Framework handles routing and state management

### Impact

Reduces multi-agent system development from days of custom coordination logic to hours of declarative team configuration.

## Proposal

### Core Architecture

Three agent types working together as a flat team:

```python
from adkx.agents import Agent, TeamAgent, LeadAgent, ExpertAgent

# Define specialists
order_specialist = ExpertAgent(
    name="order_specialist",
    model="gemini-2.5-flash",
    description="Order status, tracking, delivery inquiries",
    tools=[check_order_status],
)

refund_specialist = ExpertAgent(
    name="refund_specialist",
    model="gemini-2.5-flash",
    description="Refunds, returns, cancellations",
    tools=[process_refund],
)

# Define coordinator
lead = LeadAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="Route customer inquiries to the right specialist.",
)

# Create team
team = TeamAgent(
    name="customer_support",
    lead_agent=lead,
    experts=[order_specialist, refund_specialist],
)

# Run through ADK runner
from google.adk import Runner

runner = Runner(agent=team)
response = await runner.run("Where is my order #12345?")
```

### Agent Types

#### TeamAgent (Entry Point)

**Role**: Lightweight orchestration shell
**Responsibility**: Recognizes coordination tools and routes requests accordingly

```python
class TeamAgent(BaseAgent):
    lead_agent: LeadAgent
    experts: list[ExpertAgent]
```

**How it works**:

- Detects coordination tool responses (delegate/handoff/escalate) from events
- Routes to target agent specified in tool response
- No LLM involved - pure code-based routing
- New sessions → route to lead
- Ongoing conversations → typically route to last active agent, but may differ depending on agent configuration

#### LeadAgent (Coordinator)

**Role**: Team coordinator and planner
**Responsibility**: Delegates to specialists, synthesizes results, manages workflows

```python
class LeadAgent(Agent):
    # Inherits: model, instruction, tools
    # Auto-injected tools: delegate(), handoff(), save_plan(), execute_plan()
```

**Key behaviors**:

- See all experts' conversations and actions from session events
- Coordinate via delegate/handoff tools
- Can create workflow plans (sequential, parallel, loop)
- Receives escalations from experts

#### ExpertAgent (Specialist)

**Role**: Domain-specific expert
**Responsibility**: Execute focused tasks using domain tools

```python
class ExpertAgent(Agent):
    # Inherits: model, description, instruction, tools
    mode: Literal["interactive", "autonomous"] = "interactive"
    allowed_peers: list[str] | None = None  # Peer coordination control
    # Auto-injected tools (interactive mode): delegate(), handoff(), escalate()
```

**Mode**:

- **interactive**: Can interact with end-user and use coordination tools (handoff/escalate) to coordinate with other agents
  - Can use domain tools as well
- **autonomous**: Runs task to completion without user interaction or coordination tools, returns control automatically
  - Can still use domain-specific tools (if model supports tool use)
  - Required for models that don't support tool use
  - Used in workflow patterns (sequential, parallel) where experts execute without user interaction

**Key behaviors**:

- Focus on single domain (description defines expertise)
- Interactive mode: Coordinate with allowed peers, escalate to lead when stuck
- Autonomous mode: Complete task and return control automatically
- Workflow execution: Run in autonomous mode (sequential/parallel patterns)

### Coordination Primitives

Four coordination primitives enable all team patterns:

#### 1. Delegate (Temporary Assignment)

Agent assigns task → Target completes → Target returns control

```python
# Lead delegates to expert
delegate(target_agent_name="order_specialist", message="Check order #12345")

# Expert completes task and hands back control
handoff(target_agent_name="coordinator", message="Order found, shipped yesterday")

# Expert can also delegate to peer expert (if allowed_peers permits)
delegate(
    target_agent_name="billing_specialist", message="Need invoice for order #12345"
)
```

**Control flow**: Agent A → Agent B → Agent A (control returns)

#### 2. Handoff (Permanent Transfer)

Agent transfers control to another agent permanently

```python
# Lead hands off to expert (expert owns conversation)
handoff(target_agent_name="refund_specialist", message="Handle this refund request")

# Expert can handoff to peers (if allowed_peers permits)
handoff(target_agent_name="billing_specialist", message="Need billing adjustment")
```

**Control flow**: Agent A → Agent B → Agent C (no automatic return)

#### 3. Escalate (Expert → Lead)

Expert returns control to lead when unable to proceed

```python
# Expert escalates to lead
escalate(reason="Customer asking about product features, outside my expertise")
```

**Control flow**: Expert → Lead (always)

#### 4. Workflow Plans (Framework-Executed)

Lead creates declarative plans for sequential, parallel, or looping workflows

```python
# Sequential: flight → hotel → summary
save_plan(
    {
        "type": "sequential",
        "steps": ["flight_finder", "hotel_finder", "trip_summarizer"],
    }
)
execute_plan()  # Experts run in autonomous mode, no user interaction

# Parallel: security + performance + style reviews
save_plan(
    {
        "type": "parallel",
        "experts": ["security_reviewer", "performance_reviewer", "style_reviewer"],
    }
)
execute_plan()  # Experts run in parallel, control returns to lead when all complete

# Loop: iterate until converged
save_plan(
    {"type": "loop", "steps": ["approach_generator", "evaluator"], "max_iterations": 5}
)
execute_plan()  # Repeat until max iterations or expert signals completion
```

**Control flow**: Framework executes plan, experts run in autonomous mode

- Sequential: Execute experts in order, each completes before next starts
- Parallel: Execute experts concurrently, control returns to lead when all finish
- Loop: Repeat steps until max iterations reached

### Key Patterns Enabled

| Pattern                  | Use Case            | Expert Mode | Coordination                                                             |
| ------------------------ | ------------------- | ----------- | ------------------------------------------------------------------------ |
| **Dynamic Routing**      | Customer support    | Interactive | Lead delegates based on request type, experts can escalate               |
| **Parallel Review**      | Code review         | Autonomous  | Lead runs security + perf + style in parallel, control returns when done |
| **Sequential Pipeline**  | Trip planning       | Autonomous  | Framework executes flight → hotel → summary in sequence                  |
| **Iterative Refinement** | Research            | Autonomous  | Loop approach_generator → evaluator until score converges                |
| **Peer Collaboration**   | Workflow automation | Interactive | Experts handoff amongst themselves via coordination tools                |

### Session State Management

Coordination state tracked automatically in session events:

- **Active agent**: Determined by examining events from latest to earliest, considering each author agent's configuration
- **Coordination history**: Delegate/handoff/escalate recorded as function responses
- **Workflow state**: Plans stored in session, executed by framework
- **Custom state**: Experts can store domain state (e.g., `session.state['_authenticated_user']`)

### Implementation Highlights

1. **Weakref team references** - Bidirectional agent-team links without circular refs
1. **Event-based routing** - Extract coordination targets from function responses
1. **Type-safe plans** - Pydantic models for Sequential/Parallel/Loop/HandoffChain plans
1. **Auto-injected tools** - Lead gets delegate/handoff/save_plan/execute_plan; Experts get delegate/handoff/escalate
1. **Filtered context** - Each agent sees only relevant conversation history

## Alternatives

### Alternative 1: Single "MultiAgent" Class

One class with config-based role assignment

**Rejected because**:

- Three classes make roles explicit and type-safe
- Separate classes enable role-specific customization
- Clearer mental model (Lead vs Expert have different responsibilities)

## Open Questions

1. **Plan creation authority** - Should only lead be able to create and execute plans, or can experts also manipulate plans? If experts can create plans, what distinguishes lead from expert?
1. **Expert context scope** - When an expert is delegated to, what context should it see?
   - Only current delegation (isolated task view)?
   - All previous delegations in this session (full delegation history)?
   - Full session history including other experts' work?
   - Configurable per expert (context_scope parameter)?
1. **Parallel workflow escalation** - When one expert escalates during parallel execution:
   - Should other running experts be halted immediately or allowed to continue?
   - If an expert is mid-turn, should we wait for turn completion or interrupt?
   - How do we define "continuable" states where interruption is safe?
   - Should other experts receive context that a peer escalated?
1. **Hierarchical teams** - If we add multi-level teams later, how should escalation bubble up?

## Success Metrics

- **Community feedback**: Positive reception from early adopters, clear use cases validated
- **Developer velocity**: Reduce multi-agent setup time from days to hours
- **Code reduction**: 70%+ less coordination logic vs manual implementation
- **Maintainability**: Reusable specialists shared across 2+ teams

## Next Steps

1. **Implement core architecture** - TeamAgent, LeadAgent, ExpertAgent
1. **Build coordination primitives** - delegate(), handoff(), escalate(), workflow plans
1. **Add to samples/** - Create customer_support_team sample
1. **Document patterns** - Add team-agent guide to docs/
1. **Gather feedback** - Iterate based on early adopter usage
