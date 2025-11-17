# Copyright 2025 Wei Sun (Jack)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Function-based tool with automatic schema generation."""

from __future__ import annotations

import inspect
from typing import Any
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Union

from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionDeclaration
from pydantic import BaseModel
from pydantic import create_model
from typing_extensions import override

from .base_tool import BaseTool
from .base_tool import ToolResult


class FunctionTool(BaseTool):
  """Tool that wraps a function with automatic schema generation.

  FunctionTool automatically generates JSON schemas for parameters and output
  using Pydantic. Functions can return any type - the return value will be
  automatically wrapped in ToolResult. The tool_context parameter is optional;
  if present in the signature, it will be excluded from the parameters schema
  and passed during execution.

  Supported parameter types:
    - Primitives: str, int, float, bool
    - Collections: list, dict
    - Pydantic models (module-level or class-level only)
    - Optional/Union types: Optional[T], T | None

  Limitations:
    - Pydantic models must be defined at module or class level, NOT inside
      functions or closures. Locally-defined types cannot be resolved when
      using `from __future__ import annotations`.
    - Types must be imported at module level, NOT inside `if TYPE_CHECKING:`
      blocks. Schema generation requires runtime access to type objects.

  Example:
    ```python
    # Module-level Pydantic model (REQUIRED for schema generation)
    from pydantic import BaseModel

    class WeatherData(BaseModel):
        temperature: float
        conditions: str

    # With tool_context
    async def get_weather(
        location: str,
        unit: str = "F",
        *,
        tool_context
    ) -> WeatherData:
        '''Get current weather for a location.'''
        temp = fetch_temperature(location, unit)
        return WeatherData(temperature=temp, conditions="sunny")

    # Without tool_context
    async def convert_temp(value: float, from_unit: str, to_unit: str) -> float:
        '''Convert temperature between units.'''
        # ... conversion logic ...
        return converted_value

    # Create tool - schemas are automatically generated
    tool = FunctionTool(
        func=get_weather,
        name="get_weather",
        description="Get current weather"
    )
    ```
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      name: str | None = None,
      description: str | None = None,
  ) -> None:
    """Initialize FunctionTool with automatic schema generation.

    Args:
      func: The function to wrap. Can optionally accept tool_context as a
        keyword-only parameter. Can return any type.
      name: Tool name for function calling. If not provided, uses function name.
      description: Tool description. If not provided, uses function docstring.

    Raises:
      ValueError: If name is not provided and func has no __name__ attribute.
    """
    # Precondition: ensure we can derive a tool name
    if name is None and not hasattr(func, "__name__"):
      raise ValueError(
          "Cannot derive tool name: callable has no __name__ attribute. "
          "Please provide the 'name' parameter explicitly."
      )

    super().__init__(
        name=name or func.__name__,
        description=description or inspect.getdoc(func) or "",
    )
    self._func = func
    self._signature = inspect.signature(func)
    self._type_hints = self._extract_type_hints(func)
    self._has_tool_context = "tool_context" in self._signature.parameters

    # Generate schemas from function signature
    self._parameters_json_schema = self._build_parameters_schema()
    self._output_json_schema = self._build_output_schema()

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Get the FunctionDeclaration for this tool.

    Returns:
      FunctionDeclaration with name, description, parameters schema,
      and optional response schema.
    """
    return FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema=self._parameters_json_schema,
        response_json_schema=self._output_json_schema,
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> ToolResult:
    """Run the wrapped function with provided arguments.

    Args:
      args: LLM-filled arguments matching the parameters schema.
      tool_context: Tool execution context (passed to function if it accepts it).

    Returns:
      ToolResult with the function's return value wrapped in details.
    """
    # Convert dict arguments to match function signature types
    converted_args = self._convert_args_to_signature_types(args)

    # Add tool_context if function accepts it
    if self._has_tool_context:
      converted_args["tool_context"] = tool_context

    # Call function (async or sync)
    if inspect.iscoroutinefunction(self._func):
      result = await self._func(**converted_args)
    else:
      result = self._func(**converted_args)

    # Already a ToolResult - return as-is
    if isinstance(result, ToolResult):
      return result

    # ToolResult.details accepts: str | dict | Blob | FileData
    if isinstance(result, (str, dict)):
      return ToolResult(details=[result])

    # Pydantic model - convert to dict with JSON serialization
    if isinstance(result, BaseModel):
      return ToolResult(details=[result.model_dump(mode="json")])

    # Convert other types (int, float, bool, etc.) to string
    return ToolResult(details=[str(result)])

  def _build_parameters_schema(self) -> dict[str, Any]:
    """Build JSON schema for function parameters using Pydantic."""
    fields = self._collect_parameter_fields()

    # Empty parameters - return minimal schema
    if not fields:
      return {"type": "object", "properties": {}, "required": []}

    # Create Pydantic model for schema generation
    try:
      model = create_model(
          "Parameters", __module__=self._func.__module__, **fields
      )
      return model.model_json_schema()
    except Exception as e:
      raise TypeError(
          f"Failed to generate parameter schema for '{self._func.__name__}':"
          f" {e}\nSee FunctionTool docstring for supported types and"
          " limitations."
      ) from e

  def _build_output_schema(self) -> dict[str, Any] | None:
    """Build JSON schema for function output (return type).

    Returns:
      JSON schema dict for the function's return type, or None if no type hint.
    """
    return_type = self._type_hints.get("return")

    if return_type is None:
      return None

    # If return type is a Pydantic model, use its schema
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
      return return_type.model_json_schema()

    # Handle basic Python types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    if return_type in type_mapping:
      return type_mapping[return_type]

    # Try to use Pydantic to generate schema for the type
    try:
      model = create_model("OutputModel", value=(return_type, ...))
      schema = model.model_json_schema()
      # Extract the schema for the 'value' field
      return schema.get("properties", {}).get("value", {"type": "object"})
    except Exception:
      # Fall back to generic object
      return {"type": "object"}

  def _collect_parameter_fields(self) -> dict[str, Any]:
    """Collect Pydantic field definitions from function signature.

    Excludes tool_context parameter.
    """
    fields: dict[str, Any] = {}

    for param_name, param in self._signature.parameters.items():
      if param_name == "tool_context":
        continue

      param_type = self._type_hints.get(param_name, Any)

      # Required parameter (no default)
      if param.default is inspect.Parameter.empty:
        fields[param_name] = (param_type, ...)
      else:
        # Optional parameter with default
        fields[param_name] = (param_type, param.default)

    return fields

  def _extract_type_hints(self, func: Callable[..., Any]) -> dict[str, Any]:
    """Extract type hints from function, with fallback for local types."""
    try:
      return get_type_hints(func)
    except NameError:
      # Fallback for locally-defined types (e.g., classes in test methods)
      return func.__annotations__.copy()

  def _convert_args_to_signature_types(
      self, args: dict[str, Any]
  ) -> dict[str, Any]:
    """Convert dict arguments to match function signature types.

    Currently supports Pydantic models (including Optional[T] and T | None).

    Args:
      args: Raw arguments from LLM (all values are dicts/primitives).

    Returns:
      Arguments converted to match function signature types.
    """
    converted = {}
    for param_name, value in args.items():
      param_type = self._type_hints.get(param_name)

      # Extract the actual model type (handling Optional/Union)
      pydantic_model_type = self._extract_pydantic_model_type(param_type)

      # Convert dict to model instance if applicable
      if pydantic_model_type and isinstance(value, dict):
        converted[param_name] = pydantic_model_type(**value)
      else:
        converted[param_name] = value

    return converted

  def _extract_pydantic_model_type(
      self, param_type: Any
  ) -> type[BaseModel] | None:
    """Supports Optional[T] and T | None. Returns None if not a Pydantic model."""
    if param_type is None:
      return None

    # Handle Union types (Optional[T], T | None)
    if get_origin(param_type) is Union:
      for arg in get_args(param_type):
        if arg is not type(None) and self._is_pydantic_model(arg):
          return arg
      return None

    # Handle direct type
    return param_type if self._is_pydantic_model(param_type) else None

  def _is_pydantic_model(self, type_hint: Any) -> bool:
    """True if type_hint is a BaseModel subclass."""
    return isinstance(type_hint, type) and issubclass(type_hint, BaseModel)
