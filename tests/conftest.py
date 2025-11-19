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

"""Test configuration and shared fixtures."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
  """Set up environment variables for testing."""
  # Set fake API key for Gemini tests
  os.environ["GOOGLE_API_KEY"] = "fake-api-key-for-testing"
  yield
  # Cleanup
  if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]
