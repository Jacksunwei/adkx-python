# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0](https://github.com/Jacksunwei/adkx-python/compare/adkx-v0.1.0...adkx-v0.2.0) (2025-11-24)


### Features

* Add image generation agent example ([#23](https://github.com/Jacksunwei/adkx-python/issues/23)) ([f86ea65](https://github.com/Jacksunwei/adkx-python/commit/f86ea65b15351efeb76b08cd5e7c27ef2a7c2937))


### Documentation

* Add ADR-0002 for multi-modal function responses ([#24](https://github.com/Jacksunwei/adkx-python/issues/24)) ([79d833a](https://github.com/Jacksunwei/adkx-python/commit/79d833af3a88f4731382cb3f86e83ff9177c30bf))
* Add team-agent pattern design proposal ([#27](https://github.com/Jacksunwei/adkx-python/issues/27)) ([b2e9d57](https://github.com/Jacksunwei/adkx-python/commit/b2e9d577ef7e8f96dd5b7e3bb2f483607f217078))
* Document tested Ollama models and update defaults ([3c84b0f](https://github.com/Jacksunwei/adkx-python/commit/3c84b0f63aa5ab9321086251312f493a7dca0e55))
* Improve README and CONTRIBUTING with comprehensive content ([#26](https://github.com/Jacksunwei/adkx-python/issues/26)) ([a26e1f6](https://github.com/Jacksunwei/adkx-python/commit/a26e1f6191e85be27925290f502d34c95f5f4740))
* Reorganize README badges for clarity ([a731084](https://github.com/Jacksunwei/adkx-python/commit/a731084c632e7ee70798c525f34e1212ee6d8910))


### Chores

* Exclude CHANGELOG.md from mdformat ([cb05347](https://github.com/Jacksunwei/adkx-python/commit/cb053474392c5392cdda7541d60519f1d2eb4dc9))

## [0.1.0](https://github.com/Jacksunwei/adkx-python/compare/adkx-v0.0.3...adkx-v0.1.0) (2025-11-22)


### Features

* Add automatic ID generation for function calls and responses ([#9](https://github.com/Jacksunwei/adkx-python/issues/9)) ([4d06d49](https://github.com/Jacksunwei/adkx-python/commit/4d06d495416f17099835a74318627f0f5da2e693))
* Add BaseTool with multi-modal ToolResult ([#6](https://github.com/Jacksunwei/adkx-python/issues/6)) ([f3d6d65](https://github.com/Jacksunwei/adkx-python/commit/f3d6d65c565a431204ba0f4e3dad974415190070))
* Add FunctionTool with automatic schema generation ([11bdb9a](https://github.com/Jacksunwei/adkx-python/commit/11bdb9a5abfb6a2167cf96f16a7073c7907d60b8))
* Add Gemini LLM implementation with streaming support ([#10](https://github.com/Jacksunwei/adkx-python/issues/10)) ([b0d8cdd](https://github.com/Jacksunwei/adkx-python/commit/b0d8cddad1a0c4148991f78c51c5f443b58bc3f6))
* Add immutable BaseAgent with no hierarchy support ([#3](https://github.com/Jacksunwei/adkx-python/issues/3)) ([d774ad1](https://github.com/Jacksunwei/adkx-python/commit/d774ad19d5251883f6ecdf2de4906c0619bf524c))
* Add mdformat plugins for enhanced markdown formatting ([769f79a](https://github.com/Jacksunwei/adkx-python/commit/769f79a5ceb910ecca4e80a193046c8877c7ce4b))
* Add Ollama agent example and improve instruction formatting ([#15](https://github.com/Jacksunwei/adkx-python/issues/15)) ([cba0cb1](https://github.com/Jacksunwei/adkx-python/commit/cba0cb1d0e417508480c43edc3d8238c8746df71))
* Add Ollama LLM support with tool calling capabilities ([#14](https://github.com/Jacksunwei/adkx-python/issues/14)) ([c0b336b](https://github.com/Jacksunwei/adkx-python/commit/c0b336b03d55f97a1646d923cdb3c9fdc1f2a074))
* Add OpenAI-compatible provider architecture ([#21](https://github.com/Jacksunwei/adkx-python/issues/21)) ([61a0855](https://github.com/Jacksunwei/adkx-python/commit/61a085577de7e1f40444bc185cdec2eeff2d1a68))
* Add simplified Agent class ([#5](https://github.com/Jacksunwei/adkx-python/issues/5)) ([b0c3e66](https://github.com/Jacksunwei/adkx-python/commit/b0c3e6654e3f175faca01fcd1a1cdc1b7fb47c12))
* Add tool execution and extension points to Agent ([#7](https://github.com/Jacksunwei/adkx-python/issues/7)) ([b118f00](https://github.com/Jacksunwei/adkx-python/commit/b118f00174db9ce90f8393ee18d4af684a41bcc5))
* Allow Agent model field to accept BaseLlm instances ([#12](https://github.com/Jacksunwei/adkx-python/issues/12)) ([506c94b](https://github.com/Jacksunwei/adkx-python/commit/506c94b0b74492a6a5384ca14cfdb171ecce5e08))


### Bug Fixes

* Add mdformat-frontmatter plugin to preserve YAML frontmatter ([14e0607](https://github.com/Jacksunwei/adkx-python/commit/14e06071d5ec7050f86e7fb7e73f9302980405cc))
* Fix Gemini streaming for google-genai 1.50.1 ([#11](https://github.com/Jacksunwei/adkx-python/issues/11)) ([51d49c7](https://github.com/Jacksunwei/adkx-python/commit/51d49c7de9ea94549074cc2091f0b350b11699c6))


### Documentation

* Add ADR-0001 for tool call streaming incompatibility ([ea9b8cd](https://github.com/Jacksunwei/adkx-python/commit/ea9b8cd7e37d3499d27d94c91108ca1a7191620a))
* Add debugging tips to AGENTS.md ([cc96949](https://github.com/Jacksunwei/adkx-python/commit/cc96949ba52800c0ecd81d82ff31b04c3a8683be))
* Add informational badges to README ([#20](https://github.com/Jacksunwei/adkx-python/issues/20)) ([c2d088c](https://github.com/Jacksunwei/adkx-python/commit/c2d088c99e37cbbc3f23a979f70d441baea2e361))
* Add multi-modal FunctionResponse design documentation ([77842bb](https://github.com/Jacksunwei/adkx-python/commit/77842bb716f571fc7303adfe40758693e82ba22b))
* Add search agent example demonstrating google_search tool ([3ebd5c4](https://github.com/Jacksunwei/adkx-python/commit/3ebd5c4907b196f902cff50821e591b079845119))
* Add structured documentation with README indices ([#19](https://github.com/Jacksunwei/adkx-python/issues/19)) ([e29aa24](https://github.com/Jacksunwei/adkx-python/commit/e29aa241dde792e2cf9b3ead2f44107f71c3c7f9))
* Add unit testing principles to AGENTS.md ([193763c](https://github.com/Jacksunwei/adkx-python/commit/193763cbf5e93ce1cc10dc835ba056fee795a8b9))
* Add weather agent example with FunctionTool ([7c398ef](https://github.com/Jacksunwei/adkx-python/commit/7c398ef314e93a179ba142d56f8aac815bc3a0e2))
* Reformat metadata to YAML frontmatter in design doc ([27fa84e](https://github.com/Jacksunwei/adkx-python/commit/27fa84e1e56736850e606feb3a2c44939f88ecaf))


### Chores

* Add docs section to release-please changelog ([0575a6a](https://github.com/Jacksunwei/adkx-python/commit/0575a6a1f065f1cf42ec0675d2e76b83b42fea45))
* Add mdformat hook and format all markdown files ([672c62c](https://github.com/Jacksunwei/adkx-python/commit/672c62c2308a3837db5617ff4fd748840b82ab51))
* Improve linting and formatting infrastructure ([#8](https://github.com/Jacksunwei/adkx-python/issues/8)) ([765caf0](https://github.com/Jacksunwei/adkx-python/commit/765caf0532f502a02d90d036852b7bb83cbbedf6))
* Update autoformat.sh to use pre-commit and add mdformat hook ([064b40e](https://github.com/Jacksunwei/adkx-python/commit/064b40eb84d486e22c39c8202e8e7ed401ba1c62))

## [0.0.3](https://github.com/Jacksunwei/adkx-python/compare/adkx-v0.0.2...adkx-v0.0.3) (2025-11-16)

### Chores

- Add workflow_dispatch trigger to PyPI publish workflow ([13cfe68](https://github.com/Jacksunwei/adkx-python/commit/13cfe68983e9bb09730962e24bdfcbf4be69f793))

## [0.0.2](https://github.com/Jacksunwei/adkx-python/compare/adkx-v0.0.1...adkx-v0.0.2) (2025-11-16)

### Bug Fixes

- Remove redundant parameters from release-please workflow ([ee7d833](https://github.com/Jacksunwei/adkx-python/commit/ee7d83359cececded1e51b309bf748ee94d2abb7))

### Chores

- Add automated PyPI publishing workflow ([2033303](https://github.com/Jacksunwei/adkx-python/commit/2033303f150a5b939f4d88e4560319b6fc6eeb69))
- Add changelog sections to include chore commits ([4388c94](https://github.com/Jacksunwei/adkx-python/commit/4388c944a02ae4aa001c859d7ec9b78cf5afc60a))
- Initialize package skeleton for adkx v0.0.1 ([7634aba](https://github.com/Jacksunwei/adkx-python/commit/7634aba0760aaf343678d159cec93ea60e5addce))
- Set up release-please automation ([cc05316](https://github.com/Jacksunwei/adkx-python/commit/cc05316b2cebacfcf8b4deca65061ec7dfa7e39a))

## [Unreleased]
