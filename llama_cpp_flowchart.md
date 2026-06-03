# llama.cpp 官方代码流程图：从模型加载到推理结束

> 基于项目启动参数：
> ```
> CUDA_VISIBLE_DEVICES=2,3 ./vm_c_server \
>   /home/user/Qwen3.6-35B-A3B-MTP-IQ4_XS.gguf \
>   --kv-cache-dtype turboquant_4bit_nc \
>   --max-model-len 16384 \
>   --tensor-parallel-size 2 \
>   --gpu-memory-utilization 0.9 \
>   --served-model-name qwen3-4b-instruct \
>   --host 0.0.0.0 --port 11436 \
>   --max-num-seqs 1 \
>   --spec-method mtp \
>   --spec-draft-n-max 5
> ```
> 模型架构：Qwen3.6-35B-A3B-MTP (MoE + MTP)
> 核心源码路径：`/home/user/vm_c/llama.cpp/src/`

---

## 一、总体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        llama.cpp 总体架构                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  llama_model  │  │llama_context │  │  server (工具层)         │  │
│  │  (模型权重)   │  │ (推理上下文)  │  │  HTTP API / 推测解码    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                  │
│         ▼                 ▼                      ▼                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    ggml 后端层                                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │  │
│  │  │ CUDA BE  │ │ CUDA BE  │ │  CPU BE  │ │ Meta BE (TP)  │   │  │
│  │  │ (GPU 0)  │ │ (GPU 1)  │ │          │ │ (虚拟聚合)    │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────────┘   │  │
│  │                                                              │  │
│  │  ┌──────────────────────────────────────────────────────┐    │  │
│  │  │          ggml_backend_sched (计算图调度器)            │    │  │
│  │  └──────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    ggml 核心层                                │  │
│  │  张量定义 / 计算图构建 / 算子实现 / 内存分配                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、模型加载流程

### 2.1 整体流程图

```
main()
  │
  ▼
llama_model_load_from_file()                    [llama.cpp:323]
  │
  ├─1. gguf_init_from_file()                    解析 GGUF 文件元数据
  │     └─ 读取 magic, version, tensor count, metadata KV
  │
  ├─2. llama_model_load()                       [llama.cpp:270]
  │     │
  │     ├─2.1 llama_model_loader 构造            [llama-model-loader.h]
  │     │     └─ 解析张量信息：名称、维度、类型、偏移
  │     │     └─ 处理 mmap / direct_io 配置
  │     │     └─ 应用 kv_overrides / tensor_buft_overrides
  │     │
  │     ├─2.2 llama_model_create(ml, params)    [llama-model.cpp:293]
  │     │     └─ 根据架构类型创建具体模型实例
  │     │     └─ Qwen3.6 → LLM_ARCH_QWEN35MOE → llama_model_qwen35moe
  │     │     └─ 检查架构是否支持 SPLIT_MODE_TENSOR
  │     │
  │     ├─2.3 llama_prepare_model_devices()     [llama.cpp:125]
  │     │     └─ 详见"2.2 设备准备流程"
  │     │
  │     ├─2.4 model->load_hparams(ml)           加载超参数
  │     │     └─ n_layer, n_embd, n_head, n_expert, nextn_predict_layers...
  │     │
  │     ├─2.5 model->load_vocab(ml)             加载词表
  │     │     └─ token 文本、分数、类型、BOS/EOS/UNK...
  │     │
  │     ├─2.6 model->load_stats(ml)             加载统计信息
  │     │
  │     ├─2.7 model->load_tensors(ml)           [llama-model.cpp:1166]
  │     │     └─ 详见"2.3 张量加载流程"
  │     │
  │     └─2.8 ml.load_data()                    实际加载权重数据
  │           └─ mmap 映射或直接读取
  │
  └─3. 返回 llama_model* 指针
```

### 2.2 设备准备流程（Tensor Parallel）

```
llama_prepare_model_devices(params, model)       [llama.cpp:125]
  │
  ├─ 判断 split_mode
  │
  ├─ LLAMA_SPLIT_MODE_TENSOR 路径：
  │     │
  │     ├─ 枚举所有 GPU 设备（跳过 CPU）
  │     │     for i in ggml_backend_dev_count():
  │     │       dev = ggml_backend_dev_get(i)
  │     │       if dev.type == CPU: skip
  │     │       devs.push_back(dev)
  │     │
  │     ├─ 创建 Meta Device（虚拟聚合设备）
  │     │     model->devices.push_back({
  │     │       is_meta: true,
  │     │       dev: ggml_backend_meta_device(
  │     │         devs, n_devs,
  │     │         llama_meta_device_get_split_state,  ← 张量切分策略回调
  │     │         &model->get_split_state_ud)
  │     │     })
  │     │
  │     └─ Meta Device 聚合了所有物理 GPU
  │         对外表现为单一设备，内部自动切分
  │
  ├─ LLAMA_SPLIT_MODE_LAYER 路径：
  │     └─ 每个物理 GPU 作为独立设备加入 model->devices
  │
  └─ LLAMA_SPLIT_MODE_NONE 路径：
        └─ 仅保留 main_gpu 指定的设备
```

### 2.3 张量加载流程

```
llama_model_base::load_tensors(ml)               [llama-model.cpp:1166]
  │
  ├─1. 构建 buffer type 列表
  │     ├─ make_cpu_buft_list()                  CPU 侧 buffer 类型
  │     └─ make_gpu_buft_list(dev, split_mode)   GPU 侧 buffer 类型
  │           └─ SPLIT_MODE_ROW: 添加 split buffer type
  │           └─ 添加设备默认 buffer type
  │           └─ 添加设备额外 buffer type (如量化类型)
  │
  ├─2. 计算层分配策略
  │     ├─ i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)
  │     ├─ 按 tensor_split 或空闲显存比例分配层到设备
  │     ├─ dev_input → CPU（输入层始终在 CPU）
  │     ├─ dev_layer[il] → 按 split 分配到 GPU
  │     └─ dev_output → 按 split 分配到 GPU
  │
  ├─3. 调用 load_arch_tensors(ml)               架构特定张量加载
  │     └─ Qwen3.6 MoE: qwen35moe.cpp
  │         ├─ tok_embd (词嵌入)
  │         ├─ output_norm (输出归一化)
  │         ├─ output (LM Head)
  │         ├─ layers[0..n_layer-1]:
  │         │   ├─ attn_q, attn_k, attn_v (注意力权重)
  │         │   ├─ attn_output (注意力输出)
  │         │   ├─ ffn_gate_inp (MoE 路由)
  │         │   ├─ ffn_up_exps, ffn_gate_exps, ffn_down_exps (专家权重)
  │         │   ├─ ffn_up_exps_s, ffn_gate_exps_s, ffn_down_exps_s (共享专家)
  │         │   └─ attn_q_a_norm, attn_kv_a_norm (QK 归一化)
  │         └─ nextn (MTP 层):
  │             ├─ nextn_embd_norm
  │             ├─ nextn_attn_q, nextn_attn_k, nextn_attn_v
  │             ├─ nextn_attn_output
  │             ├─ nextn_ffn_* (MTP 专用 FFN)
  │             └─ nextn_ffn_*_shared_expert (MTP 共享专家)
  │
  └─4. 加载通用 scale 张量（NVFP4 等）
```

---

## 三、推理上下文创建流程

### 3.1 整体流程图

```
llama_init_from_model(model, params)             [llama-context.cpp:3350]
  │
  ├─ 参数校验
  │   ├─ SPLIT_MODE_TENSOR → 强制 flash_attn
  │   ├─ SPLIT_MODE_TENSOR → 禁止 KV cache 量化
  │   ├─ flash_attn + 量化 K → 检查 head_dim 对齐
  │   └─ 量化 V + 非 flash_attn → 报错
  │
  └─ new llama_context(model, params)            [llama-context.cpp:140]
        │
        ├─1. 参数初始化
        │     ├─ cparams.n_ctx = GGML_PAD(n_ctx, 256)
        │     ├─ cparams.n_ubatch = min(n_batch, n_ubatch)
        │     └─ cparams.flash_attn_type 根据条件设置
        │
        ├─2. GPU 后端初始化
        │     for dev in model.devices:
        │       backend = ggml_backend_dev_init(dev, nullptr)
        │       backends.push_back(backend)
        │     └─ Meta Device → 创建 Meta Backend（内含多个 CUDA Backend）
        │
        ├─3. ACCEL 后端初始化 (BLAS 等)
        │
        ├─4. CPU 后端初始化
        │     backend_cpu = ggml_backend_init_by_type(CPU, nullptr)
        │
        ├─5. 输出 buffer 预分配
        │     output_reserve(n_seq_max)
        │
        ├─6. 内存模块初始化
        │     model.create_memory(params_mem, cparams)
        │     └─ 详见"四、KV Cache 与显存管理流程"
        │
        ├─7. 后端 buffer type 收集
        │     ├─ 每个 backend → ggml_backend_get_default_buffer_type()
        │     └─ CPU backend → 使用首个 GPU 设备的 host buffer type
        │
        ├─8. Pipeline Parallel 判断
        │     └─ n_devices > 1 && n_gpu_layers > n_layer
        │         && SPLIT_MODE_LAYER && offload_kqv
        │
        └─9. sched_reserve()                      预留调度器资源
              │
              ├─9.1 graph_reserve() 为每种 graph_type 构建示例计算图
              │     ├─ 构造 ubatch_reserve(n_tokens/n_seqs, n_seqs)
              │     │   ├─ 分配 token[], pos[], n_seq_id[], seq_id[], output[]
              │     │   ├─ embd = nullptr（普通 context 无需 embd）
              │     │   └─ MTP context: 额外分配 embd[n_tokens * n_embd]
              │     │       └─ 确保 ubatch 结构与实际推理一致
              │     │
              │     ├─ 设置 seq_ids, output flags
              │     ├─ model->build_graph(gparams) 构建完整计算图
              │     │   └─ graph_type 由 ctx_type 决定:
              │     │       ├─ LLAMA_CONTEXT_TYPE_DECODER → LLM_GRAPH_TYPE_DECODER
              │     │       └─ LLAMA_CONTEXT_TYPE_MTP → LLM_GRAPH_TYPE_DECODER_MTP
              │     │
              │     └─ ggml_backend_sched_reserve(sched, gf)
              │         ├─ 分析计算图拓扑
              │         ├─ 为每个后端估算所需 compute buffer 大小
              │         └─ 一次性分配所有 compute buffer（运行时不再 realloc）
              │
              ├─9.2 对 Target Context: graph_reserve(DECODER)
              └─9.3 对 MTP Draft Context: graph_reserve(DECODER_MTP)
                    └─ ubatch 必须同时包含 token + embd（与实际推理一致）
```

---

## 四、KV Cache 与显存管理流程

### 4.1 KV Cache 创建流程

```
llama_model::create_memory(params_mem, cparams)  [llama-model.cpp]
  │
  ├─ 判断模型类型（标准注意力 / 循环 / 混合）
  │
  ├─ 标准注意力模型路径：
  │     │
  │     ├─ 创建 llama_kv_cache_unified
  │     │     ├─ n_layer 个 llama_kv_cache_unified_layer
  │     │     ├─ 每个 layer:
  │     │     │   ├─ k_l: [n_kv_heads, n_embd_head_k, n_ctx] (type_k)
  │     │     │   └─ v_l: [n_kv_heads, n_embd_v, n_ctx]     (type_v)
  │     │     │
  │     │     └─ type_k/type_v 由 --kv-cache-dtype 指定
  │     │         ├─ turboquant_4bit_nc: 4bit 量化 KV cache
  │     │         ├─ q8_0, q4_0 等标准量化
  │     │         └─ GGML_TYPE_F16: 默认半精度
  │     │
  │     └─ 分配设备内存
  │           ├─ 每层 KV cache 分配到对应设备
  │           └─ SPLIT_MODE_TENSOR: KV cache 按 SPLIT_AXIS_0 切分到各 GPU
  │
  └─ 显存预算计算
        ├─ 总可用显存 = gpu_memory_utilization × total_gpu_memory
        ├─ 权重占用 (已加载)
        ├─ KV cache 占用 = n_layer × 2 × n_kv_heads × head_dim × n_ctx × type_size
        ├─ 计算 buffer 预留
        └─ 剩余显存不足 → 调整 n_ctx 或报错
```

### 4.2 KV Cache 更新流程

```
llama_kv_cache_unified::update()                 [llama-kv-cache.cpp]
  │
  ├─1. 处理序列操作
  │     ├─ seq_rm: 删除指定序列的 KV cache
  │     ├─ seq_cp: 复制序列的 KV cache
  │     ├─ seq_keep: 保留指定序列
  │     └─ seq_shift: 移位 KV cache（RoPE 位置调整）
  │
  ├─2. 确定当前 batch 的 KV cache 位置
  │     ├─ 查找或分配 cell（token → position 映射）
  │     ├─ 处理重复 token（不同序列可复用）
  │     └─ 更新 cell 的 pos, seq_id, source
  │
  └─3. 返回 ubatch（包含需要计算的 token 信息）
        ├─ token ids
        ├─ positions
        ├─ seq_ids
        └─ n_tokens
```

### 4.3 KV Cache 量化（turboquant_4bit_nc）

```
turboquant_4bit_nc KV Cache 量化流程:
  │
  ├─ K Cache 量化:
  │     ├─ 原始: [n_kv_heads, n_embd_head_k, n_ctx] (F16)
  │     ├─ 量化: 按 group_size 分组量化为 4bit
  │     └─ 存储: 4bit 索引 + group scale/zp
  │
  ├─ V Cache 量化:
  │     ├─ 原始: [n_kv_heads, n_embd_v, n_ctx] (F16)
  │     ├─ 量化: 按 group_size 分组量化为 4bit
  │     └─ 存储: 4bit 索引 + group scale/zp
  │
  └─ Flash Attention 兼容性:
        ├─ 量化 K cache → 需要 head_dim % blck_size == 0
        ├─ 量化 V cache → 需要 flash_attn 启用
        └─ SPLIT_MODE_TENSOR → 禁止 KV cache 量化
```

---

## 五、推理执行流程

### 5.1 单步推理（Decode）流程

```
llama_context::decode(batch_inp)                 [llama-context.cpp]
  │
  ├─1. 输入校验
  │     ├─ batch.token || batch.embd 必须存在
  │     └─ 如果无 memory → 调用 encode()
  │
  ├─2. Batch 预处理
  │     ├─ llama_batch_reserve()
  │     └─ mctx->prepare() → 分解为 ubatch 序列
  │
  ├─3. 循环处理每个 ubatch
  │     do {
  │       ubatch = mctx->get_ubatch()
  │       │
  │       ├─3.1 memory->update(ubatch)          更新 KV cache
  │       ├─3.2 process_ubatch(ubatch, ...)      处理微批次
  │       └─3.3 mctx->next()                     下一个 ubatch
  │     } while (mctx->next())
  │
  └─4. 返回推理结果
```

### 5.2 process_ubatch 详细流程

```
llama_context::process_ubatch(ubatch, graph_type, mctx, status)
  │
  ├─1. 构建/获取计算图
  │     │
  │     ├─1.1 检查 graph_reuse: allow_reuse(gparams) 判定
  │     │     ├─ ubatch 一致性检查:
  │     │     │   ├─ equal_seqs, n_tokens, n_seq_tokens, n_seqs, n_seqs_unq 必须相同
  │     │     │   └─ token/embd 指针组合必须匹配:
  │     │     │       ├─ 两者都无 token → ✓
  │     │     │       ├─ 两者都无 embd → ✓
  │     │     │       └─ 两者都有 token 且都有 embd → ✓
  │     │     │       └─ 其他组合 → ✗（预留时无 embd 但推理时有 → 复用失败）
  │     │     │
  │     │     ├─ 每个 llm_graph_input_i 的 can_reuse() 检查:
  │     │     │   ├─ llm_graph_input_embd::can_reuse():
  │     │     │   │   └─ token/embd tensor 维度必须匹配 n_tokens
  │     │     │   └─ llm_graph_input_embd_h::can_reuse():
  │     │     │       └─ token/embd/h tensor 维度必须匹配 n_tokens
  │     │     │
  │     │     └─ 所有 input 的 can_reuse() 都通过 → 可复用
  │     │
  │     ├─1.2 可复用 → set_input(ubatch) 更新输入数据，复用已分配的 compute buffer
  │     │
  │     └─1.3 不可复用 → 重新构建:
  │           ├─ llm_graph_context 构造
  │           ├─ model->build_graph(ctx, graph_type)
  │           │   └─ 详见"5.3 计算图构建流程"
  │           ├─ ggml_backend_sched_alloc_graph(sched, gf)
  │           │   └─ ⚠️ 需要重新分配 compute buffer
  │           │   └─ 如果显存已被其他 context 占满 → 分配失败 → 崩溃!
  │           └─ ggml_build_forward_expand(gf, res->t_logits)
  │
  ├─2. 更新计算图输入
  │     ├─ 设置 token ids → inp_tokens
  │     ├─ 设置 positions → inp_pos
  │     ├─ 设置 attention metadata → inp_attn_scale, inp_out_id 等
  │     └─ 设置 KV cache 指针
  │
  ├─3. 调度器计算
  │     ├─ ggml_backend_sched_graph_compute(sched, gf)
  │     │   ├─ 分配计算 buffer
  │     │   ├─ 将计算图节点分配到各后端
  │     │   └─ 依次在各后端执行计算
  │     │
  │     └─ Meta Backend (TP) 路径:
  │         ├─ 将图切分为子图（按 AllReduce 边界）
  │         ├─ 每个子图在各 GPU 上并行执行
  │         └─ 子图间执行 AllReduce 同步
  │
  └─4. 获取输出
        ├─ res->t_logits → logits 数据
        └─ res->t_embd → embedding 数据（如适用）
```

### 5.3 计算图构建流程（Qwen3.6 MoE）

```
llama_model_qwen35moe::build_graph(ctx, graph_type)
  │
  ├─ llm_graph_context 构造
  │     ├─ 创建输入张量: inp_tokens, inp_pos, inp_out_id...
  │     ├─ 创建 KV cache 张量: k_l, v_l (每层)
  │     └─ 创建输出张量: t_logits
  │
  ├─ graph_type == LLM_GRAPH_TYPE_DECODER:
  │     └─ graph_decoder 构造
  │         │
  │         ├─ tok_embd = cb(inp_tokens → embedding lookup)
  │         ├─ cur = tok_norm(tok_embd)
  │         │
  │         ├─ for il in 0..n_layer-1:
  │         │     │
  │         │     ├─ ─── Attention Block ───
  │         │     │   ├─ cur = attn_norm(cur)
  │         │     │   ├─ Q = wq(cur), K = wk(cur), V = wv(cur)
  │         │     │   ├─ Q/K Norm: Q = q_norm(Q), K = k_norm(K)
  │         │     │   ├─ RoPE: Q = rope(Q, pos), K = rope(K, pos)
  │         │     │   ├─ Flash Attention:
  │         │     │   │   ├─ K,V → KV cache 更新
  │         │     │   │   ├─ attn_out = flash_attn(Q, K_cache, V_cache)
  │         │     │   │   └─ 支持 GQA (grouped query attention)
  │         │     │   └─ cur = wo(attn_out)
  │         │     │
  │         │     ├─ cur = cur + residual (残差连接)
  │         │     │
  │         │     ├─ ─── MoE FFN Block ───
  │         │     │   ├─ cur = ffn_norm(cur)
  │         │     │   ├─ router_logits = ffn_gate_inp(cur)
  │         │     │   ├─ topk_experts = softmax(router_logits).topk(n_expert_used)
  │         │     │   ├─ for each selected expert:
  │         │     │   │   ├─ up = silu(gate(cur)) * up(cur)
  │         │     │   │   └─ down = down(up)
  │         │     │   ├─ expert_output = weighted_sum(expert_outputs)
  │         │     │   ├─ shared_expert:
  │         │     │   │   ├─ up_s = silu(gate_s(cur)) * up_s(cur)
  │         │     │   │   └─ down_s = down_s(up_s)
  │         │     │   └─ cur = expert_output + gate * down_s
  │         │     │
  │         │     └─ cur = cur + residual (残差连接)
  │         │
  │         ├─ cur = output_norm(cur)
  │         └─ logits = output(cur) (LM Head)
  │
  └─ graph_type == LLM_GRAPH_TYPE_DECODER_MTP:
        └─ graph_mtp 构造（详见"六、MTP 推测解码流程"）
```

### 5.4 Graph 预留与复用机制

```
┌─────────────────────────────────────────────────────────────────────┐
│              Graph 预留与复用机制（核心性能优化）                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  启动时预留 (graph_reserve)                                         │
│  ════════════════════════                                           │
│  目的：在启动时一次性分配所有 compute buffer，避免推理时 realloc     │
│                                                                     │
│  graph_reserve(n_tokens, n_seqs, n_outputs, mctx)                  │
│    │                                                                │
│    ├─1. 构造 ubatch_reserve(n_tokens/n_seqs, n_seqs)               │
│    │     ├─ 分配 token[], pos[], n_seq_id[], seq_id[], output[]    │
│    │     ├─ embd = nullptr（普通 DECODER context）                  │
│    │     └─ embd = new float[n_tokens * n_embd]（MTP context）     │
│    │         └─ ⚠️ 关键：MTP 必须分配 embd，否则与推理时不一致     │
│    │                                                                │
│    ├─2. model->build_graph(gparams)                                │
│    │     └─ 根据 ubatch 内容构建完整计算图                         │
│    │     └─ ⚠️ graph 拓扑取决于 ubatch.token/embd 是否非空         │
│    │         ├─ if (ubatch.token): 只构建 token 路径               │
│    │         └─ ggml_build_forward_select: 两条路径都构建          │
│    │                                                                │
│    └─3. ggml_backend_sched_reserve(sched, gf)                      │
│          ├─ 分析计算图拓扑                                         │
│          ├─ 为每个后端估算所需 compute buffer 大小                  │
│          └─ 一次性分配所有 compute buffer                           │
│                                                                     │
│  推理时复用 (process_ubatch)                                        │
│  ════════════════════════                                           │
│  目的：复用已分配的 compute buffer，避免 realloc                    │
│                                                                     │
│  allow_reuse(other) 判定流程:                                       │
│    │                                                                │
│    ├─1. ubatch 一致性检查                                           │
│    │   ├─ equal_seqs, n_tokens, n_seq_tokens, n_seqs, n_seqs_unq  │
│    │   └─ token/embd 指针组合:                                      │
│    │       ├─ (!old.token && !new.token) → ✓                       │
│    │       ├─ (!old.embd  && !new.embd)  → ✓                       │
│    │       ├─ (old.token && new.token && old.embd && new.embd) → ✓│
│    │       └─ 其他组合 → ✗ 复用失败                                 │
│    │                                                                │
│    ├─2. 每个 llm_graph_input_i::can_reuse() 检查                   │
│    │   ├─ llm_graph_input_embd_h::can_reuse():                     │
│    │   │   ├─ (!ubatch.token) || (tokens->ne[0] == n_tokens)      │
│    │   │   ├─ (!ubatch.embd)  || (embd->ne[1] == n_tokens)       │
│    │   │   └─ (!ubatch.embd)  || (h->ne[1] == n_tokens)          │
│    │   └─ 所有 input 都通过 → 可复用                               │
│    │                                                                │
│    └─3. 结果                                                        │
│        ├─ 可复用 → set_input() 更新输入数据，零额外显存开销        │
│        └─ 不可复用 → alloc_graph() 重新分配                        │
│            └─ ⚠️ 如果显存已被占满 → 分配失败 → 崩溃!              │
│                                                                     │
│  崩溃根因（MTP 场景）                                               │
│  ══════════════════════                                             │
│  预留时: ubatch = {token: ✓, embd: ✗}                              │
│          → graph 只构建 token 路径                                  │
│          → allow_reuse 记录: token=✓, embd=✗                       │
│                                                                     │
│  推理时: ubatch = {token: ✓, embd: ✓}                              │
│          → allow_reuse 检查: (✓,✗) vs (✓,✓) → 不匹配!             │
│          → 复用失败 → alloc_graph → 显存不足 → 崩溃                │
│                                                                     │
│  修复方案                                                           │
│  ══════                                                             │
│  1. graph_reserve: MTP context 的 ubatch 必须分配 embd              │
│  2. graph_mtp: 用 ggml_build_forward_select 替代 if(ubatch.token)  │
│     → 两条路径都构建到 graph 中，运行时动态选择                     │
│     → 预留和推理的 graph 拓扑完全一致                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、MTP 推测解码流程

### 6.1 MTP 概述

MTP (Multi-Token Prediction) 推测解码利用模型内置的 MTP 头（nextn_predict_layers）
生成多个候选 token，然后通过目标模型验证，实现加速推理。

```
┌─────────────────────────────────────────────────────────────┐
│                    MTP 推测解码架构                           │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐                      │
│  │  Target CTX  │     │  Draft CTX   │                      │
│  │  (主模型)     │     │  (MTP 头)    │                      │
│  │              │     │              │                      │
│  │  完整模型     │     │  仅 MTP 层   │                      │
│  │  n_layer 层  │     │  1 层        │                      │
│  │  共享权重     │◄────┤  共享词嵌入   │                      │
│  │              │     │  共享 KV cache│                      │
│  └──────┬───────┘     └──────┬───────┘                      │
│         │                    │                               │
│         ▼                    ▼                               │
│    验证 token           生成 draft tokens                    │
│    (接受/拒绝)         (最多 spec-draft-n-max 个)            │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 MTP Context 创建流程

```
server_context 初始化                             [tools/server/server-context.cpp]
  │
  ├─1. 创建 Target Context
  │     ├─ cparams_tgt.ctx_type = LLAMA_CONTEXT_TYPE_DECODER
  │     ├─ cparams_tgt.type_k = turboquant_4bit_nc
  │     ├─ cparams_tgt.type_v = turboquant_4bit_nc
  │     └─ ctx_tgt = llama_init_from_model(model_tgt, cparams_tgt)
  │
  ├─2. 创建 MTP Draft Context
  │     ├─ cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
  │     ├─ cparams_mtp.type_k = draft.cache_type_k
  │     ├─ cparams_mtp.type_v = draft.cache_type_v
  │     ├─ cparams_mtp.n_rs_seq = 0
  │     └─ ctx_dft = llama_init_from_model(model_tgt, cparams_mtp)
  │         └─ 注意：使用同一个 model_tgt，但 context 类型不同
  │
  └─3. 配置推测解码
        ├─ params.speculative.draft.ctx_tgt = ctx_tgt
        ├─ params.speculative.draft.ctx_dft = ctx_dft
        └─ params.speculative.draft.n_max = 5 (spec-draft-n-max)
```

### 6.3 MTP 计算图构建

```
llama_model_qwen35moe::graph_mtp 构造            [models/qwen35moe.cpp]
  │
  ├─1. 输入准备（llm_graph_input_embd_h — 同时支持 token 和 embd 输入）
  │     │
  │     ├─ 创建 llm_graph_input_embd_h(n_embd)
  │     │   ├─ inp->tokens: I32 [n_tokens]     ← token ID 输入
  │     │   ├─ inp->embd:   F32 [n_embd_inp, n_tokens]  ← embedding 输入
  │     │   └─ inp->h:      F32 [n_embd, n_tokens]      ← 隐藏状态输入
  │     │
  │     ├─ Token/Embedding 路径选择（ggml_build_forward_select）:
  │     │   ├─ 路径0 (ubatch.token != nullptr):
  │     │   │   └─ tok_embd = get_rows(embed_tokens, inp->tokens)
  │     │   │       └─ 优先使用 layer.nextn.embed_tokens，否则用 model.tok_embd
  │     │   ├─ 路径1 (ubatch.token == nullptr):
  │     │   │   └─ tok_embd = inp->embd
  │     │   └─ ggml_build_forward_select(gf, [路径0, 路径1], 2, idx)
  │     │       └─ 两条路径都构建到计算图中，运行时根据 ubatch.token 选择
  │     │       └─ ⚠️ 必须使用 ggml_build_forward_select 而非 if(ubatch.token)
  │     │           否则预留时 graph 拓扑与实际推理不一致 → realloc → 崩溃
  │     │
  │     ├─ h_embd = inp->h  ← 上一层的隐藏状态（由 set_input 从 ubatch.embd 填入）
  │     │
  │     └─ res->add_input(std::move(inp))
  │
  ├─2. MTP 特有计算
  │     ├─ h_norm = rms_norm(h_embd, nextn.hnorm)    ← 隐藏状态归一化
  │     ├─ e_norm = rms_norm(tok_embd, nextn.enorm)   ← embedding 归一化
  │     ├─ concat = concat(e_norm, h_norm, dim=0)      ← 拼接两个归一化结果
  │     └─ cur = eh_proj(concat)                        ← 投影回 n_embd 维度
  │
  ├─3. MTP Attention Block（与主模型结构相同）
  │     ├─ cur = attn_norm(cur)
  │     ├─ Q = wq(cur), K = wk(cur), V = wv(cur)
  │     ├─ Q/K Norm + GQA 门控
  │     ├─ RoPE
  │     ├─ Flash Attention (使用共享 KV cache)
  │     └─ cur = wo(attn_out) + residual
  │
  ├─4. MTP MoE FFN Block（与主模型结构相同）
  │     ├─ cur = ffn_norm(cur)
  │     ├─ MoE 路由 + 专家计算
  │     │   ├─ build_moe_ffn(cur, ...)
  │     │   │   ├─ ffn_gate_inp → topk 路由
  │     │   │   ├─ 各专家: silu(gate) * up → down
  │     │   │   ├─ 共享专家: silu(gate_s) * up_s → down_s
  │     │   │   └─ expert_output + gate * shared_output
  │     │   └─ moe_out
  │     └─ cur = moe_out + residual
  │
  └─5. 输出
        ├─ 保存 h_pre_norm = cur（供下一轮 MTP draft 使用）
        ├─ cur = output_norm(cur)
        ├─ logits = output(cur) (LM Head)
        └─ ggml_build_forward_expand(gf, logits)
```

> **关键设计要点**：MTP graph 的输入处理使用 `llm_graph_input_embd_h`（而非 `llm_graph_input_embd`），
> 因为 MTP 需要同时接收 token ID（或 embedding）和上一层的隐藏状态 `h`。
> `set_input` 时，`ubatch.embd` 的数据会被同时写入 `inp->embd` 和 `inp->h` 两个 tensor。
> 
> **ggml_build_forward_select 的必要性**：MTP 推理时 ubatch 同时包含 `token` 和 `embd`，
> 如果 graph 构建时用 `if (ubatch.token)` 编译期分支，则预留时（token!=null, embd=null）
> 与推理时（token!=null, embd!=null）的 graph 拓扑不同，导致 `allow_reuse` 失败，
> 触发 `ggml_backend_sched_alloc_graph` 重新分配 compute buffer，在显存已满时崩溃。

### 6.4 MTP 推测解码执行流程

```
MTP 推测解码执行流程 (server-context.cpp + speculative.cpp)
  │
  ├─1. Prefill 阶段（首 token）
  │     ├─ ctx_tgt.decode(prompt_batch)
  │     │   └─ process() hook: 将 target 的 h_pre_norm 传递给 draft context
  │     ├─ target_token = sample(ctx_tgt, logits)
  │     └─ 将 target_token 加入输出序列
  │
  ├─2. 推测解码循环
  │     while not done:
  │       │
  │       ├─2.1 Draft 阶段：生成候选 token
  │       │     ├─ common_speculative_draft(spec)
  │       │     ├─ 使用 pending_h (上一步的 h_pre_norm) 作为 MTP 输入
  │       │     ├─ 循环生成 draft tokens:
  │       │     │   ├─ batch = [id_last, pos, embd=h_row]
  │       │     │   ├─ llama_decode(ctx_dft, batch)
  │       │     │   ├─ sample top-K → draft_token
  │       │     │   ├─ 如果 draft_token 概率 < p_min → 停止生成
  │       │     │   └─ 最多生成 n_max=5 个候选 token
  │       │     └─ 返回 draft_tokens 列表
  │       │
  │       ├─2.2 Target 验证阶段：批量验证
  │       │     ├─ 构建 batch: [current_token, draft_tokens...]
  │       │     │   └─ 每个 token 设置 logits=1
  │       │     ├─ 保存 checkpoint (KV cache 快照)
  │       │     ├─ ctx_tgt.decode(batch)
  │       │     └─ 获取每个位置的 logits
  │       │
  │       ├─2.3 贪心接受/拒绝判定
  │       │     ├─ common_sampler_sample_and_accept_n()
  │       │     ├─ for i in 0..len(draft_tokens):
  │       │     │   ├─ target_token = sample(target_logits[i])
  │       │     │   ├─ if draft_tokens[i] == target_token:
  │       │     │   │   └─ 接受 (贪心匹配成功)
  │       │     │   └─ else:
  │       │     │       └─ 拒绝，使用 target_token 替代
  │       │     │
  │       │     └─ 如果所有 draft 都被接受 → 额外采样一个 token
  │       │
  │       └─2.4 更新状态
  │             ├─ 接受的 token 加入输出序列
  │             ├─ 部分接受时:
  │             │   ├─ 恢复 checkpoint (KV cache 回滚)
  │             │   └─ 移除被拒绝 token 的 KV cache
  │             └─ common_speculative_accept() → 更新 pending_h
  │                 └─ 使用第 n_accepted 个 verify_h 行作为新的 pending_h
  │
  └─3. 返回生成的 token 序列

注意：MTP 推测解码使用**贪心匹配**而非概率比率接受：
  - 标准 speculative decoding: r < min(1, p_target / p_draft) → 随机接受
  - MTP speculative decoding: draft_token == target_sampled_token → 确定性接受
  - 原因：MTP 头与主模型共享权重，draft 质量较高，贪心匹配更高效
```

---

## 七、Tensor Parallel 流程

### 7.1 TP 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  Tensor Parallel 架构                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Meta Device (虚拟设备)                   │    │
│  │  ┌──────────────────┐  ┌──────────────────┐             │    │
│  │  │   GPU 0 (CUDA)   │  │   GPU 1 (CUDA)   │             │    │
│  │  │                  │  │                  │             │    │
│  │  │  Q[0:h/2]        │  │  Q[h/2:h]        │             │    │
│  │  │  K[0:h/2]        │  │  K[h/2:h]        │             │    │
│  │  │  V[0:h/2]        │  │  V[h/2:h]        │             │    │
│  │  │  attn_out[0:d/2] │  │  attn_out[d/2:d] │             │    │
│  │  │  FFN_up[0:f/2]   │  │  FFN_up[f/2:f]   │             │    │
│  │  │  FFN_down[0:o/2] │  │  FFN_down[o/2:o] │             │    │
│  │  │                  │  │                  │             │    │
│  │  │  NCCL Comm 0     │  │  NCCL Comm 1     │             │    │
│  │  └────────┬─────────┘  └────────┬─────────┘             │    │
│  │           │                      │                        │    │
│  │           └──────────┬───────────┘                        │    │
│  │                      │                                    │    │
│  │               AllReduce (NCCL)                            │    │
│  │                      │                                    │    │
│  └──────────────────────┼────────────────────────────────────┘    │
│                         ▼                                        │
│                  同步后的完整结果                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 张量切分策略

```
llama_meta_device_get_split_state(tensor, userdata)  [llama-model.cpp:321]
  │
  ├─ 根据张量名称匹配正则，确定切分轴：
  │
  │  ┌─────────────────────────────────────────────────────────────┐
  │  │                    张量切分策略表                            │
  │  ├──────────────────────────┬──────────────────────────────────┤
  │  │  张量类型                │  切分轴                          │
  │  ├──────────────────────────┼──────────────────────────────────┤
  │  │  attn_q.weight           │  SPLIT_AXIS_1 (列切分)          │
  │  │  attn_k.weight           │  SPLIT_AXIS_1                   │
  │  │  attn_v.weight           │  SPLIT_AXIS_1                   │
  │  │  attn_qkv.weight         │  SPLIT_AXIS_1                   │
  │  │  attn_q.bias             │  SPLIT_AXIS_0 (行切分)          │
  │  │  attn_kv.bias            │  SPLIT_AXIS_0                   │
  │  │  attn_qkv.bias           │  SPLIT_AXIS_0                   │
  │  │  attn_qk_norm.weight     │  SPLIT_AXIS_1 或 MIRRORED       │
  │  │  cache_k/v_l*            │  SPLIT_AXIS_0                   │
  │  │  attn_output.weight      │  SPLIT_AXIS_0 (行切分)          │
  │  │  attn_output.bias        │  SPLIT_AXIS_MIRRORED (镜像)     │
  │  │  ffn_up/gate.weight      │  SPLIT_AXIS_1 (列切分)          │
  │  │  ffn_down.weight         │  SPLIT_AXIS_0 (行切分)          │
  │  │  ffn_down.bias           │  SPLIT_AXIS_MIRRORED            │
  │  │  ffn_down_exps.bias      │  SPLIT_AXIS_PARTIAL (部分切分)  │
  │  │  output.weight           │  SPLIT_AXIS_1                   │
  │  │  norm / embedding        │  SPLIT_AXIS_MIRRORED (镜像)     │
  │  └──────────────────────────┴──────────────────────────────────┘
  │
  └─ 切分粒度计算:
        ├─ 按 quantization block size 对齐
        ├─ 按 head_dim 对齐 (Q/K/V)
        └─ 按 GQA 分组对齐
```

### 7.3 TP 计算图执行流程

```
ggml_backend_meta_graph_compute(backend, cgraph)  [ggml-backend-meta.cpp:1766]
  │
  ├─1. 图切分（needs_rebuild 时）
  │     ├─ 为每个 backend 复制节点映射
  │     ├─ 遍历计算图节点，确定 split_state
  │     ├─ 按 SPLIT_AXIS_PARTIAL 边界切分子图
  │     │   └─ 每个 PARTIAL 节点标志一个 AllReduce 点
  │     ├─ 优化：延迟 AllReduce（MoE 场景）
  │     │   └─ 跳过 MIRRORED 中间节点，推迟到真正需要同步时
  │     └─ 构建每个 backend 的子图
  │
  ├─2. 逐子图执行
  │     for i in 0..n_subgraphs:
  │       │
  │       ├─2.1 各 GPU 并行执行子图
  │       │     for j in 0..n_backends:
  │       │       ggml_backend_graph_compute_async(bc[j].backend, cgraph[i][j])
  │       │
  │       └─2.2 AllReduce 同步（非最后一个子图时）
  │             ├─ 优先使用 NCCL AllReduce:
  │             │   ├─ comm_allreduce(comm_ctx, nodes)
  │             │   ├─ 小张量 (<32K 元素, 2 GPU): ncclAllReduce(FP32)
  │             │   └─ 大张量: ncclAllReduce(BF16) → 转回 FP32
  │             │
  │             └─ NCCL 不可用时 → Butterfly Fallback:
  │                 ├─ 非幂2设备数: 先折叠多余设备
  │                 ├─ Butterfly reduction:
  │                 │   for offset = n/2, n/4, ..., 1:
  │                 │     for j in 0..2*offset:
  │                 │       push_data(j, j^offset)
  │                 │       ADD: node_dst += node_tmp
  │                 └─ 复制回多余设备
  │
  └─3. 返回计算结果
```

### 7.4 NCCL 通信初始化

```
ggml_backend_cuda_comm_init(backends, n_backends)  [ggml-cuda.cu:1387]
  │
  ├─ 检查所有 backend 是否为 CUDA
  │
  ├─ 收集设备 ID
  │
  ├─ 根据 GGML_CUDA_ALLREDUCE 环境变量选择模式:
  │     ├─ "nccl"     → comm_init_nccl()
  │     ├─ "internal" → comm_init_internal()
  │     ├─ "none"     → comm_init_none() (butterfly fallback)
  │     └─ 默认 (Linux): comm_init_nccl()
  │
  ├─ comm_init_nccl():
  │     ├─ ncclCommInitAll(comms, n, dev_ids)
  │     ├─ 成功 → try_allreduce = nccl_allreduce
  │     └─ 失败 → 回退到 comm_init_internal()
  │
  ├─ comm_init_internal():
  │     ├─ ggml_cuda_ar_pipeline_init(dev_ids, n)
  │     ├─ 成功 → try_allreduce = internal_allreduce
  │     └─ 失败 → 回退到 comm_init_none()
  │
  └─ comm_init_none():
        └─ try_allreduce = butterfly_allreduce
```

### 7.5 TP 中的 Attention 计算流程

```
Flash Attention in TP Mode:
  │
  ├─ 每个 GPU 持有部分 Q/K/V heads:
  │     GPU 0: Q[0:n_heads/2], K[0:n_kv_heads/2], V[0:n_kv_heads/2]
  │     GPU 1: Q[n_heads/2:], K[n_kv_heads/2:], V[n_kv_heads/2:]
  │
  ├─ Attention 计算（各 GPU 独立）:
  │     ├─ Q_local × K_local^T → attn_scores_local
  │     ├─ softmax(attn_scores_local) → attn_weights_local
  │     └─ attn_weights_local × V_local → attn_out_local
  │
  ├─ Attention Output 投影:
  │     ├─ wo 权重按 SPLIT_AXIS_0 切分
  │     ├─ 每个 GPU: partial_out = attn_out_local @ wo_local
  │     └─ partial_out 是 SPLIT_AXIS_PARTIAL → 需要 AllReduce
  │
  └─ AllReduce(attn_output):
        ├─ GPU 0: full_out = partial_out_0 + partial_out_1
        └─ GPU 1: full_out = partial_out_0 + partial_out_1
```

### 7.6 TP 中的 MoE FFN 计算流程

```
MoE FFN in TP Mode:
  │
  ├─ Router (ffn_gate_inp): MIRRORED → 各 GPU 相同结果
  │
  ├─ Expert 权重切分:
  │     ├─ ffn_up/gate: SPLIT_AXIS_1 (列切分)
  │     │   每个 GPU 持有 up/gate 的部分列
  │     ├─ ffn_down: SPLIT_AXIS_0 (行切分)
  │     │   每个 GPU 持有 down 的部分行
  │     └─ ffn_down_exps.bias: SPLIT_AXIS_PARTIAL
  │
  ├─ Expert 计算:
  │     ├─ 每个 GPU: partial_up = silu(gate_local(x)) * up_local(x)
  │     ├─ 每个 GPU: partial_down = down_local(partial_up)
  │     └─ partial_down 是 SPLIT_AXIS_PARTIAL → 需要 AllReduce
  │
  ├─ Shared Expert (同上切分策略):
  │     └─ partial_shared = down_s_local(silu(gate_s_local(x)) * up_s_local(x))
  │
  └─ AllReduce + 合并:
        ├─ full_expert = AllReduce(partial_down)
        ├─ full_shared = AllReduce(partial_shared)
        └─ output = full_expert + gate_weight * full_shared
```

---

## 八、Server 服务流程

### 8.1 Server 启动与请求处理

```
server_main()                                     [tools/server/server.cpp]
  │
  ├─1. 参数解析
  │     ├─ model path, host, port
  │     ├─ tensor-parallel-size → split_mode = TENSOR
  │     ├─ kv-cache-dtype → type_k/type_v
  │     ├─ max-model-len → n_ctx
  │     ├─ gpu-memory-utilization
  │     ├─ spec-method = mtp → 启用 MTP 推测解码
  │     └─ spec-draft-n-max = 5
  │
  ├─2. 模型加载
  │     └─ llama_model_load_from_file(path, params)
  │
  ├─3. 上下文创建
  │     ├─ ctx_tgt (Target Context)
  │     └─ ctx_dft (MTP Draft Context)
  │
  ├─4. 启动 HTTP 服务
  │     └─ 监听 host:port
  │
  └─5. 请求处理循环
        ├─ POST /v1/chat/completions
        │   ├─ 解析请求 (prompt, max_tokens, temperature...)
        │   ├─ 创建 sampling context
        │   ├─ Prefill: 编码 prompt
        │   ├─ Decode: 推测解码循环
        │   └─ 返回生成的文本
        │
        └─ POST /v1/completions
            └─ 类似流程
```

---

## 九、关键数据结构关系

```
┌─────────────────────────────────────────────────────────────┐
│                     核心数据结构关系                         │
│                                                              │
│  llama_model                                                 │
│  ├─ hparams: llama_hparams        (超参数)                  │
│  ├─ vocab:  llama_vocab           (词表)                    │
│  ├─ devices: vector<llama_device> (设备列表)                │
│  │   └─ [is_meta, ggml_backend_dev_t]                       │
│  ├─ layers:  vector<llama_layer>  (各层权重)                │
│  │   ├─ wq, wk, wv, wo            (注意力权重)              │
│  │   ├─ ffn_gate_inp              (MoE 路由)               │
│  │   ├─ ffn_up/gate/down_exps     (专家权重)                │
│  │   ├─ ffn_up/gate/down_exps_s   (共享专家)                │
│  │   └─ nextn_*                    (MTP 层权重)             │
│  ├─ output, output_norm           (LM Head)                 │
│  └─ tok_embd                      (词嵌入)                  │
│                                                              │
│  llama_context                                               │
│  ├─ model: llama_model&           (模型引用)                │
│  ├─ memory: llama_kv_cache_unified (KV Cache)               │
│  ├─ backends: vector<backend>      (后端列表)                │
│  │   ├─ Meta Backend (TP)                                   │
│  │   │   └─ 内含多个 CUDA Backend                            │
│  │   ├─ ACCEL Backend                                        │
│  │   └─ CPU Backend                                          │
│  ├─ sched: ggml_backend_sched      (调度器)                 │
│  ├─ buf_output: buffer             (输出 buffer)            │
│  └─ sampling: sampler_state        (采样状态)                │
│                                                              │
│  ggml_backend_sched                                          │
│  ├─ 分配计算图节点到各后端                                   │
│  ├─ 管理计算 buffer 的分配和复用                             │
│  └─ 执行计算图的异步调度                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、完整端到端流程总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                    端到端流程：从启动到推理                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 启动阶段                                                        │
│  ══════════                                                         │
│  main()                                                             │
│    ├─ 解析命令行参数                                                 │
│    ├─ 设置 CUDA_VISIBLE_DEVICES=2,3                                 │
│    └─ 配置 tensor-parallel-size=2 → split_mode=TENSOR               │
│                                                                     │
│  2. 模型加载阶段                                                    │
│  ══════════════                                                     │
│  llama_model_load_from_file()                                       │
│    ├─ GGUF 元数据解析                                                │
│    ├─ 识别架构: Qwen3.6 MoE                                         │
│    ├─ 创建 Meta Device (GPU0 + GPU1)                                │
│    ├─ 加载超参数 / 词表                                              │
│    ├─ 加载张量 (按 TP 策略切分到各 GPU)                              │
│    │   ├─ Q/K/V: 列切分 (每 GPU 持有部分 heads)                     │
│    │   ├─ attn_out: 行切分 (每 GPU 持有部分输出维度)                 │
│    │   ├─ FFN up/gate: 列切分                                       │
│    │   ├─ FFN down: 行切分                                          │
│    │   └─ Norm/Embedding: 镜像 (每 GPU 完整副本)                    │
│    └─ mmap 映射权重数据                                              │
│                                                                     │
│  3. 上下文创建阶段                                                  │
│  ════════════════                                                   │
│  llama_init_from_model()                                            │
│    ├─ 创建 Target Context                                           │
│    │   ├─ 初始化 Meta Backend (含 CUDA Backend × 2)                 │
│    │   ├─ 初始化 CPU Backend                                        │
│    │   ├─ 创建 KV Cache (turboquant_4bit_nc)                        │
│    │   ├─ 分配计算 buffer                                           │
│    │   └─ sched_reserve() 预留资源                                  │
│    │                                                                 │
│    └─ 创建 MTP Draft Context                                        │
│        ├─ ctx_type = LLAMA_CONTEXT_TYPE_MTP                         │
│        ├─ 共享同一 model 的 MTP 层权重                              │
│        └─ 独立的 KV Cache (与 Target 共享数据)                      │
│                                                                     │
│  4. 推理阶段                                                        │
│  ══════════                                                         │
│  4.1 Prefill                                                        │
│    ├─ 编码 prompt tokens                                            │
│    ├─ 构建 Decoder 计算图                                           │
│    ├─ Meta Backend 执行:                                            │
│    │   ├─ 切分子图 (按 AllReduce 边界)                              │
│    │   ├─ 各 GPU 并行计算子图                                       │
│    │   ├─ 子图间 NCCL AllReduce 同步                                │
│    │   └─ 重复直到所有子图完成                                      │
│    └─ 采样首 token                                                  │
│                                                                     │
│  4.2 Speculative Decode Loop                                        │
│    ├─ Draft 阶段:                                                   │
│    │   ├─ MTP Context 生成 5 个候选 token                           │
│    │   └─ 每个 draft token: MTP 图计算 → 采样                       │
│    │                                                                 │
│    ├─ Target 验证阶段:                                              │
│    │   ├─ 批量编码 [current + draft_tokens]                         │
│    │   ├─ Meta Backend 执行 (同 Prefill)                            │
│    │   └─ 获取每个位置的 logits                                     │
│    │                                                                 │
│    ├─ 接受/拒绝判定:                                                │
│    │   ├─ 比较目标概率与 draft 概率                                  │
│    │   ├─ 接受的 token 加入输出                                     │
│    │   └─ 拒绝时从修正分布采样                                      │
│    │                                                                 │
│    └─ 更新 KV Cache:                                                │
│        ├─ 保留接受 token 的 cache                                   │
│        └─ 移除拒绝 token 的 cache                                   │
│                                                                     │
│  5. 输出阶段                                                        │
│  ══════════                                                         │
│  ├─ Detokenize: token ids → 文本                                    │
│  ├─ 流式/非流式返回                                                  │
│  └─ 释放资源                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键源文件索引

| 文件路径 | 核心功能 |
|---------|---------|
| `src/llama.cpp` | 模型加载入口、设备准备 |
| `src/llama-context.cpp` | 推理上下文、decode/process_ubatch、graph_reserve |
| `src/llama-model.cpp` | 模型基类、张量加载、切分策略 |
| `src/llama-model-loader.h` | GGUF 加载器、张量元数据 |
| `src/llama-kv-cache.h` | KV Cache 管理 |
| `src/llama-hparams.h` | 超参数定义 |
| `src/llama-graph.h` | 计算图构建上下文、allow_reuse 判定 |
| `src/llama-graph.cpp` | build_inp_embd (ggml_build_forward_select)、input can_reuse |
| `src/llama-batch.cpp` | ubatch_reserve 构造、batch 分割 |
| `src/models/qwen35moe.cpp` | Qwen3.6 MoE 模型实现、MTP graph (graph_mtp) |
| `ggml/src/ggml-backend-meta.cpp` | Meta Backend (TP 核心) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA Backend、NCCL AllReduce |
| `tools/server/server-context.cpp` | Server 上下文、MTP 配置 |
| `common/speculative.cpp` | 推测解码框架 |