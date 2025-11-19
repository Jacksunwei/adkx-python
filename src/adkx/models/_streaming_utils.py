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

"""Package-private utilities for streaming response merging.

This module provides common utilities for merging streaming LLM responses.
These functions handle the accumulation and merging of response parts from
streaming API calls.
"""

from __future__ import annotations

from itertools import groupby

from google.adk.models.llm_response import LlmResponse
from google.genai import types


def merge_response_parts(responses: list[LlmResponse]) -> list[types.Part]:
  """Merge all content parts, combining consecutive parts of same type.

  Args:
    responses: List of LlmResponse objects from streaming.

  Returns:
    List of merged Part objects.
  """
  all_parts = collect_all_parts(responses)
  if not all_parts:
    return []

  return group_and_merge_parts(all_parts)


def collect_all_parts(responses: list[LlmResponse]) -> list[types.Part]:
  """Collect all parts from all responses.

  Args:
    responses: List of LlmResponse objects.

  Returns:
    List of all Part objects from all responses.
  """
  all_parts = []
  for resp in responses:
    if resp.content and resp.content.parts:
      all_parts.extend(resp.content.parts)
  return all_parts


def group_and_merge_parts(parts: list[types.Part]) -> list[types.Part]:
  """Group consecutive parts by type and merge them.

  Consecutive text parts are merged into one text part.
  Consecutive thought parts are merged into one thought part.
  Other parts (function_call, function_response, etc.) are kept as-is.

  Args:
    parts: List of Part objects to merge.

  Returns:
    List of merged Part objects.
  """

  def get_part_type(part: types.Part) -> str:
    if part.thought:
      return "thought"
    if part.text:
      return "text"
    return "other"

  merged = []
  for part_type, group in groupby(parts, key=get_part_type):
    group_parts = list(group)

    if part_type == "text":
      merged.append(merge_text_parts(group_parts))
    elif part_type == "thought":
      merged.append(merge_thought_parts(group_parts))
    else:
      merged.extend(group_parts)

  return merged


def merge_text_parts(parts: list[types.Part]) -> types.Part:
  """Merge consecutive text parts into one.

  Args:
    parts: List of Part objects with text content.

  Returns:
    Single Part with concatenated text.
  """
  merged_text = "".join(p.text for p in parts if p.text)
  return types.Part(text=merged_text)


def merge_thought_parts(parts: list[types.Part]) -> types.Part:
  """Merge consecutive thought parts into one.

  Args:
    parts: List of Part objects with thought content.

  Returns:
    Single Part with concatenated thought text.
  """
  merged_text = "".join(p.text for p in parts if p.text)
  return types.Part(thought=True, text=merged_text)
