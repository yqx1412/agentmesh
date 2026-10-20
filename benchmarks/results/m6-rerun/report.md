## Overall

| | single | multi |
|---|---|---|
| Runs passed | 34/48 (71%) | 47/48 (98%) |
| Passed, every cited page fetched | 23/48 (48%) | 46/48 (96%) |
| Checks passed | 60/78 (77%) | 77/78 (99%) |
| Median time (s) | 31 | 44 |
| Mean time (s) | 39 | 51 |
| Mean tokens | 5,501 | 20,438 |
| Mean model calls | 2.2 | 6.3 |
| Mean tool calls | 1.7 | 3.2 |
| Mean A2A tasks | 1.0 | 1.4 |
| Runs citing an unfetched page | 13 of 13 | 1 of 18 |
| Failures | partial 6, wrong 8 | partial 1 |

## By category

| Category | single passed | single median s | single mean tokens | multi passed | multi median s | multi mean tokens |
|---|---|---|---|---|---|---|
| research | 10/15 (67%) | 26 | 3,991 | 14/15 (93%) | 43 | 28,669 |
| analysis | 12/12 (100%) | 39 | 5,105 | 12/12 (100%) | 42 | 6,045 |
| knowledge | 6/9 (67%) | 26 | 4,859 | 9/9 (100%) | 23 | 5,543 |
| verification | 0/3 (0%) | 50 | 7,775 | 3/3 (100%) | 49 | 39,973 |
| composite | 6/9 (67%) | 37 | 8,432 | 9/9 (100%) | 82 | 34,293 |

## By task

| Task | single passed | single median s | multi passed | multi median s | multi plan |
|---|---|---|---|---|---|
| r-qwen-license | 2/3 | 12 | 2/3 | 67 | web_research > claim_verification (x3) |
| r-a2a-origin | 3/3 | 20 | 3/3 | 39 | web_research (x2); web_research > claim_verification |
| r-ollama-port | 3/3 | 28 | 3/3 | 42 | web_research (x3) |
| r-llama-draft | 2/3 | 38 | 3/3 | 44 | web_research (x3) |
| r-mcp-definition | 0/3 | 27 | 3/3 | 39 | web_research (x3) |
| a-kv-cache | 3/3 | 38 | 3/3 | 39 | computation (x3) |
| a-weights | 3/3 | 76 | 3/3 | 44 | computation (x3) |
| a-latency-stats | 3/3 | 52 | 3/3 | 49 | computation (x3) |
| a-throughput | 3/3 | 27 | 3/3 | 39 | computation (x3) |
| k-radix-analogy | 3/3 | 24 | 3/3 | 22 | knowledge_lookup (x3) |
| k-peak-timing | 3/3 | 26 | 3/3 | 24 | knowledge_lookup (x3) |
| k-table-doubling | 0/3 | 31 | 3/3 | 26 | knowledge_lookup (x3) |
| v-three-claims | 0/3 | 50 | 3/3 | 49 | claim_verification (x3) |
| c-qwen-kv | 1/3 | 80 | 3/3 | 84 | web_research > computation (x3) |
| c-ollama-ratio | 2/3 | 37 | 3/3 | 91 | web_research > web_research > computation (x3) |
| c-peak-speedup | 3/3 | 28 | 3/3 | 71 | knowledge_lookup > knowledge_lookup > computation (x3) |

Per task, multi passed more often on 6, less often on 0 and equally on 10.

## Failures and citation problems

| System | Task | Rep | Failure | Detail |
|---|---|---|---|---|
| multi | r-qwen-license | 1 | cites | ; cites unfetched https://developer.nvidia.com/downloads/assets/ace/model_card/qwen3-8b-instruct.pdf |
| multi | r-qwen-license | 2 | partial | answer lacks all of ['apache']; got 'The Qwen3-8B model has a native context length of 32,768 tokens and can be ex...' |
| single | r-qwen-license | 0 | partial | answer !~ /32[,. ]?768\|32\s?k\b/; got 'The Qwen3-8B model is released under the **Apache 2.0 license**, as stated on...'; cites unfetched https://llm-explorer.com/model/Qwen%2FQwen3-8B |
| single | r-qwen-license | 1 | cites | ; cites unfetched https://modelscope.cn/models/qwen3-8b/summary, https://www.alibabacloud.com/blog/qwen3-8b-technical-details |
| single | r-qwen-license | 2 | cites | ; cites unfetched https://modelscope.cn/models/qwen3-8b/summary, https://www.alibabacloud.com/blog/qwen3-8b-technical-details |
| single | r-a2a-origin | 0 | cites | ; cites unfetched https://github.com/a2aproject |
| single | r-a2a-origin | 1 | cites | ; cites unfetched https://github.com/a2aproject |
| single | r-a2a-origin | 2 | cites | ; cites unfetched https://github.com/a2aproject/A2A |
| single | r-ollama-port | 0 | cites | ; cites unfetched https://ethannguyen2k.github.io/blog/ollama-server.html |
| single | r-ollama-port | 1 | cites | ; cites unfetched https://convly.ai/ollama-port-configuration-guide |
| single | r-ollama-port | 2 | cites | ; cites unfetched https://mljourney.com/how-to-configure-ollama-environment-variables-complete-guide |
| single | r-llama-draft | 0 | cites | ; cites unfetched https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf/speculative-decoding |
| single | r-llama-draft | 1 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The command-line option in llama.cpp's llama-server that sets the draft model..."; cites unfetched https://github.com/ggerganov/llama.cpp |
| single | r-llama-draft | 2 | cites | ; cites unfetched https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf/speculative-decoding |
| single | r-mcp-definition | 0 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | r-mcp-definition | 1 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | r-mcp-definition | 2 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | k-table-doubling | 0 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | k-table-doubling | 1 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | k-table-doubling | 2 | partial | no timestamp in 18:30-22:00; found 4 timestamp(s) |
| single | v-three-claims | 0 | wrong | answer !~ /FALSE:\s*\[?\s*(claim\s*)?2\s*\]?\s*\.?\s*$/; got 'FALSE: 2,3' |
| single | v-three-claims | 1 | wrong | answer !~ /FALSE:\s*\[?\s*(claim\s*)?2\s*\]?\s*\.?\s*$/; got 'FALSE: 2,3' |
| single | v-three-claims | 2 | wrong | answer !~ /FALSE:\s*\[?\s*(claim\s*)?2\s*\]?\s*\.?\s*$/; got 'FALSE: 2,3' |
| single | c-qwen-kv | 0 | partial | answer lacks 4.5; got 'The required information about the head dimension (head_dim) for Qwen3-8B is ...' |
| single | c-qwen-kv | 1 | wrong | answer lacks 36; got "The required information about Qwen3-8B's architecture (number of layers, KV ..."; answer lacks 4.5; got "The required information about Qwen3-8B's architecture (number of layers, KV ..." |
| single | c-qwen-kv | 2 | cites | ; cites unfetched https://developer.nvidia.com/downloads/assets/ace/model_card/qwen3-8b-instruct.pdf |
| single | c-ollama-ratio | 1 | partial | answer lacks 4096; got 'RATIO: 8.0' |
