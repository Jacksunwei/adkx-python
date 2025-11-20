## Python Tips

### General Python Best Practices

- **Constants:** Use immutable global constant collections (tuple, frozenset, immutabledict) to avoid hard-to-find bugs. Prefer constants over wild string/int literals, especially for dictionary keys, pathnames, and enums.
- **Naming:** Name mappings like `value_by_key` to enhance readability in lookups (e.g., `item = item_by_id[id]`).
- **Readability:** Use f-strings for concise string formatting, but use lazy-evaluated `%`-based templates for logging. Use `repr()` or `pprint.pformat()` for human-readable debug messages. Use `_` as a separator in numeric literals to improve readability.
- **Comprehensions:** Use list, set, and dict comprehensions for building collections concisely.
- **Iteration:** Iterate directly over containers without indices. Use `enumerate()` when you need the index, `dict.items()` for keys and values, and `zip()` for parallel iteration.
- **Built-ins:** Leverage built-in functions like `all()`, `any()`, `reversed()`, `sum()`, etc., to write more concise and efficient code.
- **Flattening Lists:** Use `itertools.chain.from_iterable()` to flatten a list of lists efficiently without unnecessary copying.
- **String Methods:** Use `startswith()` and `endswith()` with a tuple of strings to check for multiple prefixes or suffixes at once.
- **Decorators:** Use decorators to add common functionality (like logging, timing, caching) to functions without modifying their core logic. Use `functools.wraps()` to preserve the original function's metadata.
- **Context Managers:** Use `with` statements and context managers (from `contextlib` or custom classes with `__enter__`/`__exit__`) to ensure resources are properly initialized and torn down, even in the presence of exceptions.
- **Else Clauses:** Utilize the `else` clause in `try/except` blocks (runs if no exception), and in `for/while` loops (runs if the loop completes without a `break`) to write more expressive and less error-prone code.
- **Single Assignment:** Prefer single-assignment form (assign to a variable once) over assign-and-mutate to reduce bugs and improve readability. Use conditional expressions where appropriate.
- **Equality vs. Identity:** Use `is` or `is not` for singleton comparisons (e.g., `None`, `True`, `False`). Use `==` for value comparison.
- **Object Comparisons:** When implementing custom classes, be careful with `__eq__`. Return `NotImplemented` for unhandled types. Consider edge cases like subclasses and hashing. Prefer using `attrs` or `dataclasses` to handle this automatically.
- **Hashing:** If objects are equal, their hashes must be equal. Ensure attributes used in `__hash__` are immutable. Disable hashing with `__hash__ = None` if custom `__eq__` is implemented without a proper `__hash__`.
- **`__init__()` vs. `__new__()`:** `__new__()` creates the object, `__init__()` initializes it. For immutable types, modifications must happen in `__new__()`.
- **Default Arguments:** NEVER use mutable default arguments. Use `None` as a sentinel value instead.
- **`__add__()` vs. `__iadd__()`:** `x += y` (in-place add) can modify the object in-place if `__iadd__` is implemented (like for lists), while `x = x + y` creates a new object. This matters when multiple variables reference the same object.
- **Properties:** Use `@property` to create getters and setters only when needed, maintaining a simple attribute access syntax. Avoid properties for computationally expensive operations or those that can fail.
- **Modules for Namespacing:** Use modules as the primary mechanism for grouping and namespacing code elements, not classes. Avoid `@staticmethod` and methods that don't use `self`.
- **Argument Passing:** Python is call-by-value, where the values are object references (pointers). Assignment binds a name to an object. Modifying a mutable object through one name affects all names bound to it.
- **Keyword/Positional Arguments:** Use `*` to force keyword-only arguments and `/` to force positional-only arguments. This can prevent argument transposition errors and make APIs clearer, especially for functions with multiple arguments of the same type.
- **Type Hinting:** Annotate code with types to improve readability, debuggability, and maintainability. Use abstract types from `collections.abc` for container annotations (e.g., `Sequence`, `Mapping`, `Iterable`). Annotate return values, including `None`. Choose the most appropriate abstract type for function arguments and return types.
- **`NewType`:** Use `typing.NewType` to create distinct types from primitives (like `int` or `str`) to prevent argument transposition and improve type safety.
- **`__repr__()` vs. `__str__()`:** Implement `__repr__()` for unambiguous, developer-focused string representations, ideally evaluable. Implement `__str__()` for human-readable output. `__str__()` defaults to `__repr__()`.
- **F-string Debug:** Use `f"{expr=}"` for concise debug printing, showing both the expression and its value.

### Libraries and Tools

- **`collections.Counter`:** Use for efficiently counting hashable objects in an iterable.
- **`collections.defaultdict`:** Useful for avoiding key checks when initializing dictionary values, e.g., appending to lists.
- **`heapq`:** Use `heapq.nlargest()` and `heapq.nsmallest()` for efficiently finding the top/bottom N items. Use `heapq.merge()` to merge multiple sorted iterables.
- **`attrs` / `dataclasses`:** Use these libraries to easily define simple classes with boilerplate methods like `__init__`, `__repr__`, `__eq__`, etc., automatically generated.
- **NumPy:** Use NumPy for efficient array computing, element-wise operations, math functions, filtering, and aggregations on numerical data.
- **Pandas:** When constructing DataFrames row by row, append to a list of dicts and call `pd.DataFrame()` once to avoid inefficient copying. Use `TypedDict` or `dataclasses` for intermediate row data.
- **Flags:** Use libraries like `argparse` or `click` for command-line flag parsing. Access flag values in a type-safe manner.
- **Serialization:** For cross-language serialization, consider JSON (built-in), Protocol Buffers, or msgpack. For Python serialization with validation, use `pydantic` for runtime validation and automatic (de)serialization, or `cattrs` for performance-focused (de)serialization with `dataclasses` or `attrs`.
- **Regular Expressions:** Use `re.VERBOSE` to make complex regexes more readable with whitespace and comments. Choose the right method (`re.search`, `re.fullmatch`). Avoid regexes for simple string checks (`in`, `startswith`, `endswith`). Compile regexes used multiple times with `re.compile()`.
- **Caching:** Use `functools.lru_cache` with care. Prefer immutable return types. Be cautious when memoizing methods, as it can lead to memory leaks if the instance is part of the cache key; consider `functools.cached_property`.
- **Pickle:** Avoid using `pickle` due to security risks and compatibility issues. Prefer JSON, Protocol Buffers, or msgpack for serialization.
- **Multiprocessing:** Be aware of potential issues with `multiprocessing` on some platforms, especially concerning `fork`. Consider alternatives like threads (`concurrent.futures.ThreadPoolExecutor`) or `asyncio` for I/O-bound tasks.
- **Debugging:** Use `IPython.embed()` or `pdb.set_trace()` to drop into an interactive shell for debugging. Use visual debuggers if available. Log with context, including inputs and exception info using `logging.exception()` or `exc_info=True`.
- **Property-Based Testing & Fuzzing:** Use `hypothesis` for property-based testing that generates test cases automatically. For coverage-guided fuzzing, consider `atheris` or `python-afl`.

### Testing

- **Assertions:** Use pytest's native `assert` statements with informative expressions. Pytest automatically provides detailed failure messages showing the values involved. Add custom messages with `assert condition, "helpful message"` when the expression alone isn't clear.
- **Custom Assertions:** Write reusable helper functions (not methods) for repeated complex checks. Use `pytest.fail("message")` to explicitly fail a test with a custom message.
- **Parameterized Tests:** Use `@pytest.mark.parametrize` to reduce duplication when running the same test logic with different inputs. This is more idiomatic than the `parameterized` library.
- **Fixtures:** Use pytest fixtures (with `@pytest.fixture`) for test setup, teardown, and dependency injection. Fixtures are cleaner than class-based setup methods and can be easily shared across tests.
- **Mocking:** Use `mock.create_autospec()` with `spec_set=True` to create mocks that match the original object's interface, preventing typos and API mismatch issues. Use context managers (`with mock.patch(...)`) to manage mock lifecycles and ensure patches are stopped. Prefer injecting dependencies via fixtures over patching.
- **Asserting Mock Calls:** Use `mock.ANY` and other matchers for partial argument matching when asserting mock calls (e.g., `assert_called_once_with`).
- **Temporary Files:** Use pytest's `tmp_path` and `tmp_path_factory` fixtures for creating isolated and automatically cleaned-up temporary files/directories. These are preferred over the `tempfile` module in pytest tests.
- **Avoid Randomness:** Do not use random number generators to create inputs for unit tests. This leads to flaky, hard-to-debug tests. Instead, use deterministic, easy-to-reason-about inputs that cover specific behaviors.
- **Test Invariants:** Focus tests on the invariant behaviors of public APIs, not implementation details.
- **Test Organization:** Prefer simple test functions over class-based tests unless you need to share fixtures across multiple test methods in a class. Use descriptive test names that explain the behavior being tested.

### Error Handling

- **Re-raising Exceptions:** Use a bare `raise` to re-raise the current exception, preserving the original stack trace. Use `raise NewException from original_exception` to chain exceptions, providing context. Use `raise NewException from None` to suppress the original exception's context.
- **Exception Messages:** Always include a descriptive message when raising exceptions.
- **Converting Exceptions to Strings:** `str(e)` can be uninformative. `repr(e)` is often better. For full details including tracebacks and chained exceptions, use functions from the `traceback` module (e.g., `traceback.format_exception(e)`, `traceback.format_exc()`).
- **Terminating Programs:** Use `sys.exit()` for expected terminations. Uncaught non-`SystemExit` exceptions should signal bugs. Avoid functions that cause immediate, unclean exits like `os.abort()`.
- **Returning None:** Be consistent. If a function can return a value, all paths should return a value (use `return None` explicitly). Bare `return` is only for early exit in conceptually void functions (annotated with `-> None`).

## Debugging Tips

### Working with Third-Party Libraries

- **Read Source Code in .venv:** When docs are unclear, read the actual implementation in `.venv/lib/python*/site-packages/`. Understanding `google.adk.models.base_llm.BaseLlm` was critical for implementing Gemini correctly.
- **Study Contracts:** Look for abstract methods, docstrings, and type hints that define expected behavior. Example: `BaseLlm.generate_content_async()` specifies streaming vs non-streaming contracts.
- **Compare Implementations:** Examine existing implementations for patterns. Studying `google.adk.models.google_llm.GoogleLlm` revealed metadata merging strategies.

### Async and Type Checking

- **AsyncGenerator Types:** Must annotate as `AsyncGenerator[YieldType, SendType]`. Example: `async def gen() -> AsyncGenerator[int, None]:`.
- **Don't Await Generators:** Use `async for x in generator()`, NOT `async for x in await generator()`. Async generators return async iterators directly.
- **Type Ignore for SDK Gaps:** Use `# type: ignore[attr-defined]` when SDKs have incomplete stubs but runtime code is correct.

### Streaming Implementation

- **Partial vs Final:** Mark streaming chunks as `partial=True`, final accumulated response as `partial=False, turn_complete=True`.
- **Accumulate Then Merge:** Collect all partial responses first, then create final merged response. Don't merge incrementally.
- **Merge Strategies by Field:**

* Lists (citations, parts): Concatenate all
* Last wins (usage, finish_reason): Use final chunk
* Dict merge (custom_metadata): Later overrides earlier

- **Group Consecutive Parts:** Use `itertools.groupby()` to group and merge consecutive text or thought parts cleanly.

### Testing Async Code

- **Mock Async Generators:** Create actual async generator functions:

```python
async def mock_gen(*args, **kwargs):
    for item in items:
        yield item


mock_client.method = mock_gen
```

- **Helper to Collect Results:**

```python
async def collect_all(async_gen, *args):
    results = []
    async for item in async_gen(*args):
        results.append(item)
    return results
```

- **Use Fixtures:** Create reusable fixtures for common objects (models, requests) to reduce test duplication.

### Common Pitfalls

- **Part Type Detection:** For thought parts, check `part.thought` (boolean flag); text is still in `part.text`, not a separate attribute.
- **Missing Type Annotations:** Always annotate function parameters and returns. Helps type checkers catch bugs.
- **Explicit Type Hints for Collections:** Use `results: list[Response] = []` not just `results = []` to enable type checking on list operations.

### Debugging Strategies

- **Start Simple:** Implement non-streaming first, verify, then add streaming.
- **Test Edge Cases:** Empty inputs, single items, multiple items, mixed types.
- **Use repr():** Debug complex objects with `repr(obj)` or `pprint.pformat(vars(obj))`.
- **Compare with Reference:** When reimplementing, compare behavior with original using same inputs.
- **Run Both pytest and mypy:** Each catches different bug classes.
