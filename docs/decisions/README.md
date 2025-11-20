# Architecture Decision Records (ADRs)

Immutable records of significant architectural decisions.

## What is an ADR?

Documents **what** decision was made, **why**, and **consequences**. ADRs are immutable once accepted. To reverse a decision, create a new superseding ADR.

See [TEMPLATE.md](TEMPLATE.md) for format.

## Current ADRs

### Proposed

*(None - first will be ADR-0001)*

### Accepted

*(None)*

### Deprecated

*(None)*

### Superseded

*(None)*

______________________________________________________________________

## Creating a New ADR

**1. Number it**: Next sequential number (start with 0001)

**2. Copy template**: `cp TEMPLATE.md 0001-decision-name.md`

**3. Fill content**: Context → Decision → Consequences

**4. Get review**: PR, discuss, merge when consensus reached

## When to Create ADRs

**Create for:**

- Architectural decisions, technology selections
- Significant design patterns, hard-to-reverse decisions

**Skip for:**

- Trivial details, easily reversible decisions
- Experiments (use `experiments/`)

## Status Updates

ADRs are immutable once accepted. Only update status line:

- **Proposed** → **Accepted** → **Deprecated** → **Superseded**

See [docs/README.md](../README.md) for document lifecycle.
