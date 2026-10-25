## Overall

| | single | multi |
|---|---|---|
| Runs passed | 33/48 (69%) | 48/48 (100%) |
| Passed, every cited page fetched | 26/48 (54%) | 48/48 (100%) |
| Checks passed | 61/78 (78%) | 78/78 (100%) |
| Median time (s) | 31 | 40 |
| Mean time (s) | 38 | 49 |
| Mean tokens | 5,472 | 14,635 |
| Mean model calls | 2.1 | 5.5 |
| Mean tool calls | 1.4 | 2.7 |
| Mean A2A tasks | 1.0 | 1.2 |
| Runs citing an unfetched page | 13 of 13 | 0 of 20 |
| Failures | partial 10, wrong 5 | none |

## By category

| Category | single passed | single median s | single mean tokens | multi passed | multi median s | multi mean tokens |
|---|---|---|---|---|---|---|
| research | 7/15 (47%) | 28 | 4,385 | 15/15 (100%) | 36 | 17,391 |
| analysis | 12/12 (100%) | 43 | 4,646 | 12/12 (100%) | 47 | 8,449 |
| knowledge | 6/9 (67%) | 23 | 4,812 | 9/9 (100%) | 22 | 5,905 |
| verification | 3/3 (100%) | 33 | 8,774 | 3/3 (100%) | 20 | 9,450 |
| composite | 5/9 (56%) | 39 | 7,947 | 9/9 (100%) | 95 | 28,747 |

## By task

| Task | single passed | single median s | multi passed | multi median s | multi plan |
|---|---|---|---|---|---|
| r-qwen-license | 1/3 | 27 | 3/3 | 43 | web_research (x3) |
| r-a2a-origin | 3/3 | 28 | 3/3 | 37 | web_research (x3) |
| r-ollama-port | 3/3 | 27 | 3/3 | 36 | web_research (x3) |
| r-llama-draft | 0/3 | 42 | 3/3 | 31 | web_research (x3) |
| r-mcp-definition | 0/3 | 27 | 3/3 | 29 | web_research (x3) |
| a-kv-cache | 3/3 | 49 | 3/3 | 44 | computation (x3) |
| a-weights | 3/3 | 76 | 3/3 | 48 | computation (x3) |
| a-latency-stats | 3/3 | 42 | 3/3 | 46 | computation (x3) |
| a-throughput | 3/3 | 33 | 3/3 | 58 | computation (x3) |
| k-radix-analogy | 3/3 | 18 | 3/3 | 22 | knowledge_lookup (x3) |
| k-peak-timing | 3/3 | 24 | 3/3 | 20 | knowledge_lookup (x3) |
| k-table-doubling | 0/3 | 31 | 3/3 | 26 | knowledge_lookup (x3) |
| v-three-claims | 3/3 | 33 | 3/3 | 20 | claim_verification (x3) |
| c-qwen-kv | 2/3 | 76 | 3/3 | 99 | web_research > computation (x3) |
| c-ollama-ratio | 0/3 | 43 | 3/3 | 90 | web_research > computation (x3) |
| c-peak-speedup | 3/3 | 31 | 3/3 | 94 | knowledge_lookup > computation (x3) |

Per task, multi passed more often on 6, less often on 0 and equally on 10.

## Failures and citation problems

| System | Task | Rep | Failure | Detail |
|---|---|---|---|---|
| single | r-qwen-license | 0 | cites | ; cites unfetched https://modelscope.cn/models/qwen3-8b/summary, https://www.alibabacloud.com/blog/announcing-qwen3-8b-and-qwen3-8b-instruct_532354 |
| single | r-qwen-license | 1 | partial | answer !~ /32[,. ]?768\|32\s?k\b/; got 'The Qwen3-8B model is released under the **Apache-2.0 license**, as stated on...'; cites unfetched https://localllms.dev/llm/qwenqwen3-8b, https://contextwindows.dev/models/qwen3-8b |
| single | r-qwen-license | 2 | partial | answer !~ /32[,. ]?768\|32\s?k\b/; got 'The Qwen3-8B model is released under the **Apache 2.0 license**, as stated on...'; cites unfetched https://en.wikipedia.org/wiki/Qwen, https://www.datalearner.com/en/ai-models/pretrained-models/Qwen3-8B |
| single | r-a2a-origin | 0 | cites | ; cites unfetched https://github.com/a2aproject/A2A, https://www.linuxfoundation.org/press, https://www.forbes.com/sites |
| single | r-a2a-origin | 1 | cites | ; cites unfetched https://agentica.wiki/articles/agent2agent-protocol, https://github.com/a2aproject, https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability |
| single | r-a2a-origin | 2 | cites | ; cites unfetched https://github.com/a2aproject/A2A, https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents |
| single | r-ollama-port | 0 | cites | ; cites unfetched https://ai-ollama.github.io/ollama-port-11434.html |
| single | r-ollama-port | 1 | cites | ; cites unfetched https://convly.ai/ollama-port-configuration-guide |
| single | r-ollama-port | 2 | cites | ; cites unfetched https://ai-ollama.github.io/ollama-port-11434.html |
| single | r-llama-draft | 0 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The command-line option for setting the draft model in llama.cpp's llama-serv..."; cites unfetched https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| single | r-llama-draft | 1 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The documentation for llama.cpp's speculative decoding is discussed in the [l..."; cites unfetched https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md, https://github.com/ggml-org/llama.cpp/tree/master/docs |
| single | r-llama-draft | 2 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The command-line option for setting the draft model in llama.cpp's llama-serv..."; cites unfetched https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md, https://llama.app/docs/cli |
| single | r-mcp-definition | 0 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | r-mcp-definition | 1 | partial | answer !~ /usb[- ]?c/; got 'MCP stands for **Model Context Protocol** in the context of AI applications. ...'; cites unfetched https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro |
| single | r-mcp-definition | 2 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | k-table-doubling | 0 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | k-table-doubling | 1 | partial | no timestamp in 18:30-22:00; found 2 timestamp(s) |
| single | k-table-doubling | 2 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | c-qwen-kv | 0 | partial | answer lacks 36; got 'The required parameters (number of layers, KV heads, and head dimension) for ...' |
| single | c-ollama-ratio | 0 | partial | answer lacks 4096; got 'RATIO: 8.0' |
| single | c-ollama-ratio | 1 | partial | answer lacks 4096; got 'RATIO: 8.0' |
| single | c-ollama-ratio | 2 | partial | answer lacks 4096; got 'RATIO: 8.0' |
