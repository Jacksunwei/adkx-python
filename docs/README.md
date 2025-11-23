# ADKX Documentation

Welcome to the ADKX documentation! This directory contains research, design proposals, and architectural decisions.

## Directory Structure

```
docs/
├── research/       # Technical research & analysis
├── design/         # RFCs & design proposals
├── decisions/      # Architecture Decision Records (ADRs)
└── guides/         # How-to guides (future)
```

## Quick Navigation

### 📊 Research

Exploratory research, comparisons, and technical investigations. These documents inform future design decisions.

**Browse by topic**: See [research/README.md](research/README.md) for full index

### 📝 Design Proposals (RFCs)

Formal proposals for significant features or changes.

**Browse all**: See [design/README.md](design/README.md)

**Format**: Problem → Motivation → Proposal → Alternatives

### ✅ Architecture Decisions (ADRs)

Immutable records of finalized architectural decisions. Numbered sequentially, never edited after merging.

**Browse all**: See [decisions/README.md](decisions/README.md)

**Format**: Context → Decision → Consequences

## Document Lifecycle

```
1. Research Phase
   └─> docs/research/[topic]/descriptive-name-YYYY-MM.md
       ↓
2. Proposal Phase (when ready)
   └─> docs/design/feature-name-YYYY-MM.md
       ↓
3. Decision Phase (after approval)
   └─> docs/decisions/NNNN-decision-name.md
       ↓
4. Implementation
   └─> Code merged to main
```

## Document Types

### Research Documents

**Purpose**: Investigate topics, compare options, document findings

**Characteristics**:

- Can be incomplete or evolving
- May contain "dead ends" (what doesn't work)
- Dated but not frozen
- Informal, exploratory tone

### RFCs (Request for Comments)

**Purpose**: Propose concrete changes with clear problem statements

**Characteristics**:

- Structured format
- Requires review/approval
- Snapshot in time (doesn't update after approval)
- Links to ADR after decision

### ADRs (Architecture Decision Records)

**Purpose**: Record finalized decisions and their context

**Characteristics**:

- Immutable (never edit, create new ADR to reverse)
- Numbered sequentially (ADR-0001, ADR-0002, etc.)
- Short and focused (1-2 pages max)
- Shows consequences (positive, negative, neutral)

## Contributing

### Adding Research

```bash
# Choose appropriate topic folder or create new one
docs/research/[topic-name]/your-analysis.md

# Update index
docs/research/README.md
```

### Proposing a Design

```bash
# Create RFC with date suffix (month precision)
docs/design/feature-name-YYYY-MM.md

# Get team review before implementation
```

### Recording a Decision

```bash
# Create ADR with next sequential number
docs/decisions/NNNN-decision-name.md

# Follow ADR template
# Link to original RFC if applicable
```

## Related Directories

- **[samples/](../samples/)** - Runnable code samples (committed)
- **[experiments/](../experiments/)** - Local-only scratch work (gitignored)
- **[tests/](../tests/)** - Test suite

## Questions?

- **"Where should my document go?"** - See document types above
- **"What's the ADR template?"** - See [decisions/TEMPLATE.md](decisions/TEMPLATE.md)
