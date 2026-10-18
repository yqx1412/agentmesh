## Overall

| | single | multi |
|---|---|---|
| Runs passed | 35/48 (73%) | 39/48 (81%) |
| Passed, every cited page fetched | 23/48 (48%) | 37/48 (77%) |
| Checks passed | 58/75 (77%) | 64/75 (85%) |
| Median time (s) | 27 | 45 |
| Mean time (s) | 34 | 59 |
| Mean tokens | 4,707 | 16,032 |
| Mean model calls | 2.0 | 6.0 |
| Mean tool calls | 1.3 | 3.3 |
| Mean A2A tasks | 1.0 | 1.4 |
| Runs citing an unfetched page | 14 of 14 | 2 of 19 |
| Failures | partial 5, wrong 8 | partial 6, wrong 3 |

## By category

| Category | single passed | single median s | single mean tokens | multi passed | multi median s | multi mean tokens |
|---|---|---|---|---|---|---|
| research | 10/15 (67%) | 25 | 3,596 | 13/15 (87%) | 43 | 18,880 |
| analysis | 12/12 (100%) | 36 | 5,105 | 12/12 (100%) | 53 | 5,498 |
| knowledge | 6/9 (67%) | 23 | 4,861 | 9/9 (100%) | 19 | 5,690 |
| verification | 1/3 (33%) | 12 | 3,851 | 3/3 (100%) | 43 | 30,623 |
| composite | 6/9 (67%) | 38 | 6,160 | 2/9 (22%) | 122 | 30,808 |

## By task

| Task | single passed | single median s | multi passed | multi median s | multi plan |
|---|---|---|---|---|---|
| r-qwen-license | 3/3 | 12 | 3/3 | 45 | web_research (x3) |
| r-a2a-origin | 3/3 | 25 | 3/3 | 81 | web_research > claim_verification > knowledge_write (x3) |
| r-ollama-port | 3/3 | 26 | 3/3 | 38 | web_research (x3) |
| r-llama-draft | 1/3 | 28 | 3/3 | 43 | web_research (x3) |
| r-mcp-definition | 0/3 | 24 | 1/3 | 35 | web_research (x3) |
| a-kv-cache | 3/3 | 36 | 3/3 | 61 | computation (x3) |
| a-weights | 3/3 | 71 | 3/3 | 60 | computation (x3) |
| a-latency-stats | 3/3 | 47 | 3/3 | 35 | computation (x3) |
| a-throughput | 3/3 | 25 | 3/3 | 47 | computation (x3) |
| k-radix-analogy | 3/3 | 17 | 3/3 | 20 | knowledge_lookup (x3) |
| k-peak-timing | 3/3 | 23 | 3/3 | 19 | knowledge_lookup (x3) |
| k-table-doubling | 0/3 | 28 | 3/3 | 18 | knowledge_lookup (x3) |
| v-three-claims | 1/3 | 12 | 3/3 | 43 | claim_verification (x3) |
| c-qwen-kv | 1/3 | 58 | 0/3 | 122 | web_research > computation (x3) |
| c-ollama-ratio | 2/3 | 35 | 0/3 | 246 | web_research > web_research > computation > computation (x3) |
| c-peak-speedup | 3/3 | 26 | 2/3 | 67 | knowledge_lookup > computation (x3) |

Per task, multi passed more often on 4, less often on 3 and equally on 9.

## Failures and citation problems

| System | Task | Rep | Failure | Detail |
|---|---|---|---|---|
| multi | r-qwen-license | 1 | cites | ; cites unfetched https://atomic.chat/models/qwen3-8b |
| multi | r-qwen-license | 2 | cites | ; cites unfetched https://atomic.chat/models/qwen3-8b |
| single | r-qwen-license | 0 | cites | ; cites unfetched https://localllms.dev/llm/qwenqwen3-8b, https://huggingface.co/Qwen/Qwen3-8B |
| single | r-qwen-license | 1 | cites | ; cites unfetched https://modelscope.cn/models/qwen3-8b/summary, https://www.alibabacloud.com/blog/qwen3-8b-technical-details |
| single | r-qwen-license | 2 | cites | ; cites unfetched https://modelscope.cn/models/qwen3-8b/summary, https://www.alibabacloud.com/blog/qwen3-8b-technical-details |
| single | r-a2a-origin | 0 | cites | ; cites unfetched https://github.com/a2aproject |
| single | r-a2a-origin | 1 | cites | ; cites unfetched https://github.com/a2aproject |
| single | r-a2a-origin | 2 | cites | ; cites unfetched https://github.com/a2aproject |
| single | r-ollama-port | 0 | cites | ; cites unfetched https://ethannguyen2k.github.io/blog/ollama-server.html |
| single | r-ollama-port | 1 | cites | ; cites unfetched https://convly.ai/ollama-port-configuration-guide, https://mljourney.com/how-to-configure-ollama-environment-variables-complete-guide |
| single | r-ollama-port | 2 | cites | ; cites unfetched https://mljourney.com/how-to-configure-ollama-environment-variables-complete-guide |
| single | r-llama-draft | 0 | cites | ; cites unfetched https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf/speculative-decoding |
| single | r-llama-draft | 1 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The command-line option in llama.cpp's llama-server that sets the draft model..."; cites unfetched https://github.com/ggerganov/llama.cpp |
| single | r-llama-draft | 2 | wrong | answer !~ /(?<![\w-])-md\b\|--model-draft\|--spec-draft-model/; got "The command-line option for setting the draft model in llama.cpp's `llama-ser..."; cites unfetched https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md |
| multi | r-mcp-definition | 1 | partial | answer !~ /usb[- ]?c/; got 'MCP stands for Model Context Protocol, an open-source standard for connecting...' |
| multi | r-mcp-definition | 2 | partial | answer !~ /usb[- ]?c/; got 'MCP stands for Model Context Protocol, an open-source standard for connecting...' |
| single | r-mcp-definition | 0 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | r-mcp-definition | 1 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | r-mcp-definition | 2 | wrong | answer lacks all of ['model context protocol']; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...'; answer !~ /usb[- ]?c/; got 'The term "MCP" is not mentioned in the provided lecture notes from MIT 6.006,...' |
| single | k-table-doubling | 0 | partial | no timestamp in 18:30-22:00; found 2 timestamp(s) |
| single | k-table-doubling | 1 | partial | no timestamp in 18:30-22:00; found 2 timestamp(s) |
| single | k-table-doubling | 2 | partial | no timestamp in 18:30-22:00; found 2 timestamp(s) |
| single | v-three-claims | 0 | wrong | answer !~ /FALSE:\s*\[?\s*(claim\s*)?2\s*\]?\s*\.?\s*$/; got 'FALSE: 1,2,3' |
| single | v-three-claims | 2 | wrong | answer !~ /FALSE:\s*\[?\s*(claim\s*)?2\s*\]?\s*\.?\s*$/; got 'FALSE: 1,2,3' |
| multi | c-qwen-kv | 0 | partial | answer lacks 4.5; got 'The KV-cache size for Qwen3-8B cannot be computed because the context length ...' |
| multi | c-qwen-kv | 1 | partial | answer lacks 4.5; got 'The fp16 KV-cache size for Qwen3-8B at its full native context length is **90...' |
| multi | c-qwen-kv | 2 | partial | answer lacks 4.5; got 'The KV-cache size for Qwen3-8B cannot be computed because the context length ...' |
| single | c-qwen-kv | 0 | partial | answer lacks 4.5; got "The required information about Qwen3-8B's architecture is not fully available..." |
| single | c-qwen-kv | 1 | wrong | answer lacks 36; got 'The required information about the number of layers, KV heads, and head dimen...'; answer lacks 4.5; got 'The required information about the number of layers, KV heads, and head dimen...' |
| single | c-qwen-kv | 2 | cites | ; cites unfetched https://developer.nvidia.com/downloads/assets/ace/model_card/qwen3-8b-instruct.pdf |
| multi | c-ollama-ratio | 0 | wrong | answer lacks 4096; got 'RATIO: 0.125'; answer !~ /RATIO:\s*8(\.0+)?\s*$/; got 'RATIO: 0.125' |
| multi | c-ollama-ratio | 1 | wrong | answer lacks 4096; got 'RATIO: 0.125'; answer !~ /RATIO:\s*8(\.0+)?\s*$/; got 'RATIO: 0.125' |
| multi | c-ollama-ratio | 2 | partial | answer !~ /RATIO:\s*8(\.0+)?\s*$/; got "Ollama's default context window size is 4096 tokens according to its official..." |
| single | c-ollama-ratio | 1 | partial | answer lacks 4096; got 'RATIO: 8.0' |
| single | c-ollama-ratio | 2 | cites | ; cites unfetched https://docs.ollama.com/faq, https://huggingface.co/Qwen/Qwen3-8B |
| multi | c-peak-speedup | 0 | wrong | answer lacks 13000; got 'The task could not be completed because the required time data for the Θ(n) a...' |
