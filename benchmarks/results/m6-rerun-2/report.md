## Overall

| | single | multi |
|---|---|---|
| Runs passed | 36/48 (75%) | 48/48 (100%) |
| Passed, every cited page fetched | 26/48 (54%) | 48/48 (100%) |
| Checks passed | 63/78 (81%) | 78/78 (100%) |
| Median time (s) | 34 | 45 |
| Mean time (s) | 42 | 50 |
| Mean tokens | 6,066 | 15,895 |
| Mean model calls | 2.1 | 5.6 |
| Mean tool calls | 1.5 | 2.8 |
| Mean A2A tasks | 1.0 | 1.2 |
| Runs citing an unfetched page | 12 of 12 | 0 of 19 |
| Failures | partial 7, wrong 5 | none |

## By category

| Category | single passed | single median s | single mean tokens | multi passed | multi median s | multi mean tokens |
|---|---|---|---|---|---|---|
| research | 10/15 (67%) | 32 | 4,387 | 15/15 (100%) | 40 | 22,914 |
| analysis | 12/12 (100%) | 45 | 4,646 | 12/12 (100%) | 50 | 8,449 |
| knowledge | 6/9 (67%) | 25 | 4,812 | 9/9 (100%) | 23 | 5,905 |
| verification | 3/3 (100%) | 35 | 8,774 | 3/3 (100%) | 21 | 9,450 |
| composite | 5/9 (56%) | 44 | 11,109 | 9/9 (100%) | 96 | 26,263 |

## By task

| Task | single passed | single median s | multi passed | multi median s | multi plan |
|---|---|---|---|---|---|
| r-qwen-license | 3/3 | 32 | 3/3 | 49 | web_research (x3) |
| r-a2a-origin | 3/3 | 32 | 3/3 | 39 | web_research (x3) |
| r-ollama-port | 3/3 | 30 | 3/3 | 40 | web_research (x3) |
| r-llama-draft | 1/3 | 40 | 3/3 | 40 | web_research (x3) |
| r-mcp-definition | 0/3 | 20 | 3/3 | 32 | web_research (x3) |
| a-kv-cache | 3/3 | 52 | 3/3 | 47 | computation (x3) |
| a-weights | 3/3 | 81 | 3/3 | 51 | computation (x3) |
| a-latency-stats | 3/3 | 45 | 3/3 | 49 | computation (x3) |
| a-throughput | 3/3 | 35 | 3/3 | 62 | computation (x3) |
| k-radix-analogy | 3/3 | 19 | 3/3 | 23 | knowledge_lookup (x3) |
| k-peak-timing | 3/3 | 25 | 3/3 | 22 | knowledge_lookup (x3) |
| k-table-doubling | 0/3 | 33 | 3/3 | 23 | knowledge_lookup (x3) |
| v-three-claims | 3/3 | 35 | 3/3 | 21 | claim_verification (x3) |
| c-qwen-kv | 2/3 | 86 | 3/3 | 103 | web_research > computation (x3) |
| c-ollama-ratio | 0/3 | 44 | 3/3 | 90 | web_research > computation (x3) |
| c-peak-speedup | 3/3 | 33 | 3/3 | 96 | knowledge_lookup > computation (x3) |

Per task, multi passed more often on 5, less often on 0 and equally on 11.

## Failures and citation problems

| System | Task | Rep | Failure | Detail |
|---|---|---|---|---|
| single | r-qwen-license | 0 | cites | ; cites unfetched https://modelscope.cn/models/qwen3-8b/summary, https://www.alibabacloud.com/blog/announcing-qwen3-8b-and-qwen3-8b-instruct_532354 |
| single | r-qwen-license | 1 | cites | ; cites unfetched https://localllms.dev/llm/qwenqwen3-8b, https://open-llm-radar.com/archive/legacy/models/qwen3-8b, https://llm-explorer.com/model/Qwen/Qwen3-8B-Base |
| single | r-qwen-license | 2 | cites | ; cites unfetched https://huggingface.co/Qwen/Qwen3-8B, https://localllms.dev/llm/qwenqwen3-8b |
| single | r-a2a-origin | 0 | cites | ; cites unfetched https://github.com/a2aproject/A2A, https://www.forbes.com/sites/janakirammsv/2026/08/19/agent2agent-joins-the-agentic-ai-foundation-alongside-mcp |
| single | r-a2a-origin | 1 | cites | ; cites unfetched https://agentica.wiki/articles/agent2agent-protocol, https://cryptothreads.io/insights/a2a-protocol-linux-foundation |
| single | r-a2a-origin | 2 | cites | ; cites unfetched https://github.com/a2aproject/A2A, https://www.forbes.com/sites/janakirammsv/2026/08/19/agent2agent-joins-the-agentic-ai-foundation-alongside-mcp |
| single | r-ollama-port | 0 | cites | ; cites unfetched https://convly.ai/ollama-port-configuration-guide |
| single | r-ollama-port | 1 | cites | ; cites unfetched http://127.0.0.1:11434, https://convly.ai/ollama-port-configuration-guide |
| single | r-ollama-port | 2 | cites | ; cites unfetched https://mljourney.com/how-to-configure-ollama-environment-variables-complete-guide |
| single | r-llama-draft | 0 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got 'The provided search results do not explicitly mention a command-line option f...'; cites unfetched https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| single | r-llama-draft | 1 | cites | ; cites unfetched https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf/speculative-decoding |
| single | r-llama-draft | 2 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The command-line option for setting the draft model in llama.cpp's llama-serv..."; cites unfetched https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| single | r-mcp-definition | 0 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | r-mcp-definition | 1 | wrong | answer lacks all of ['model context protocol']; got 'The provided web search results do not explicitly define what MCP stands for ...'; answer !~ /usb[- ]?c/; got 'The provided web search results do not explicitly define what MCP stands for ...' |
| single | r-mcp-definition | 2 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | k-table-doubling | 0 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | k-table-doubling | 1 | partial | no timestamp in 18:30-22:00; found 2 timestamp(s) |
| single | k-table-doubling | 2 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | c-qwen-kv | 1 | partial | answer lacks 36; got 'The fp16 KV-cache size for Qwen3-8B at its full native context length is 4.5 ...' |
| single | c-ollama-ratio | 0 | partial | answer lacks 4096; got 'RATIO: 8.0' |
| single | c-ollama-ratio | 1 | partial | answer lacks 4096; got 'RATIO: 8.0' |
| single | c-ollama-ratio | 2 | partial | answer lacks 4096; got 'RATIO: 8.0' |
