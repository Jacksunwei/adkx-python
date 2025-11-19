# ADR-NNNN: [Short Title of Decision]

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded by ADR-XXXX

## Context

What is the issue that we're seeing that is motivating this decision or change?

Include:
- Current situation
- Problem/challenge faced
- Constraints (technical, organizational, time)
- Forces at play (competing concerns)

## Decision

What is the change that we're actually proposing or have agreed to?

Be specific:
- What will we do?
- What technology/pattern/approach will we use?
- Why this option over alternatives?

Use active voice:
- ✅ "We will use the OpenAI SDK for all providers"
- ❌ "The OpenAI SDK could be used"

## Consequences

What becomes easier or more difficult because of this change?

### Positive
- Benefit 1
- Benefit 2
- ...

### Negative / Trade-offs
- Limitation 1
- Cost 2
- ...

### Neutral
- Implication 1 (neither clearly good nor bad)
- Change 2
- ...

---

## Notes

### When to Create an ADR

Create an ADR when:
- Making architectural decisions that affect system structure
- Choosing between multiple viable technical approaches
- Establishing standards or patterns for the codebase
- Making decisions that will be hard to reverse later

**Don't create ADRs for:**
- Trivial implementation details
- Decisions easily reversible with low cost
- Temporary experiments (use experiments/ directory)

### ADR Lifecycle

1. **Proposed** - Under review, not yet approved
2. **Accepted** - Approved and being/been implemented
3. **Deprecated** - Decision no longer recommended but still in use
4. **Superseded** - Replaced by a newer ADR (link to it)

### Immutability

Once an ADR is **Accepted** and merged:
- ✅ Add clarifying notes at the bottom
- ✅ Update status (Accepted → Deprecated → Superseded)
- ❌ Never change the Decision or Consequences sections
- ❌ Never delete ADRs

To reverse a decision: Create a new ADR that supersedes the old one.

### Numbering

ADRs are numbered sequentially starting from 0001:
- `0001-first-decision.md`
- `0002-second-decision.md`
- ...
- `0042-answer-to-everything.md`

Check existing ADRs to find the next number.

### Linking

Reference ADRs in code, docs, and pull requests:
```markdown
<!-- See ADR-0003 for why we chose this approach -->
```

Link to related RFCs:
```markdown
## References
- RFC: [OpenAI Multi-Provider Support](../design/openai-multi-provider-2025-11.md)
```
