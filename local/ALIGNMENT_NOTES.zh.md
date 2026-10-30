# CTC 对齐说明：标点处理与对齐块构造

本文档描述 `ctc_tts` 中文本-语音对齐阶段的处理逻辑：标点如何被处理，以及每一个对齐块是如何构造的。

## 1. 适用范围

- 核心实现：`ctc_model/ctc_align.py`
- 调用方：`local/VoiceAssistant/align_data.py`（VoiceAssistant-400K）、`local/data_prep_multi_proc.py`（LibriSpeech）
- 单条样本输入：一个 wav 文件 + 对应文本。VoiceAssistant 的对齐文本是
  `item["units_label"]`（音素/unit 串），随后由 tokenizer 编码。
- 单条样本输出（写入 `aligned_{split}.json`）：
  `units`（用空格连接的音素符号串，含标点）、`seqs`（展平后的块 token 序列），
  当开启 `--write_textgrid` 时另额外输出 `.TextGrid` 文件。

入口函数 `align_text_logits_and_speech_tokens(...)` 的调用链：

```
文本 ──encode_text_with_separators──> TextTokenizationResult
     ──forced_align_ctc_logits────> 原始 CTC 路径
     ──postprocess_alignment_path─> token_path / token_lengths
     ──align_text_tokens_to_speech_tokens──> 每个文本 token 的语音区间
     ──build_interleaved_sequence / build_unit_speech_blocks──> 序列
```

## 2. 对齐用到的配置项

取自 `config/monophone/single-speaker/o21_single_speaker_ctc_tts_l.json`：

| 配置项 | 值 | 含义 |
| --- | --- | --- |
| `encoder.tokens_per_second` | `25` | CTC 编码器帧率（每帧 40 ms） |
| `nac.tokens_per_second` | `75` | WavTokenizer 语音 token 速率（每 token 13.33 ms） |
| `nac.codebook_size` | `4096` | 语音码本大小；同时作为文本 token 的偏移量 |
| `data.separators` | `{" ":4196, ",":4197, ".":4198, "!":4199, "?":4200}` | 分隔符字符 -> token id |
| `data.unit_token_start` / `_end` | `4096` / `4195` | 文本 unit（音素）token id 范围 |
| `data.separator_token_start` / `_end` | `4196` / `4200` | 分隔符 token id 范围 |
| `data.eob` | `4201` | 块结束 token |
| `data.pad` | `4202` | 填充 token |
| `data.eos` | `4203` | 句子结束 token |
| `data.block_type` | `"biword"` | 块粒度：当前词（并带下一词的前瞻） |

派生值：

- 文本偏移 `_get_text_offset` = `nac.codebook_size` = **4096**（回退到 `data.unit_token_start`）。
  tokenizer id `t` 变为“偏移后”的 id `t + 4096`，使文本 token、语音 token、分隔符 token 共用同一 id 空间。
- 时间轴比例 `_get_ratio` = `nac.tokens_per_second / encoder.tokens_per_second`
  = `75 / 25` = **3.0**（一个编码器帧 = 三个语音 token）。

## 3. 标点是如何处理的

### 3.1 按分隔符切分文本

`split_text_by_config_separators(text, config)` 用配置中的分隔符集合切分原始文本。
由于 `" "`（空格）本身也是分隔符，切分得到的**每个 segment 就是一个单词**。
它返回：

- `segments`：去掉分隔符后的单词串，内部空白被规整，空串被丢弃。
- `separators`：每个 segment 后面所跟的分隔符字符（仅对确实后面有分隔符的 segment 记录）。
  若最后一个 segment 没有尾随分隔符，则 `separators` 比 `segments` 少一项。

示例：`"ab c."` -> `segments = ["ab", "c"]`，`separators = [" ", "."]`。

### 3.2 编码产生三条并行的序列

`encode_text_with_separators` 对每个 segment 调用 `tokenizer.encode(segment)`，并构造：

| 字段 | 内容 | 是否含标点 |
| --- | --- | --- |
| `ctc_token_ids` | 各 segment 编码结果的拼接（原始 tokenizer id） | **不含** |
| `shifted_token_ids` | `ctc_token_ids` + 偏移 4096 | **不含** |
| `shifted_token_ids_with_separators` | 偏移后的文本 token，并在每个 segment 之后插入分隔符 token id | **含** |
| `segment_lengths` | 每个 segment 的 token 数 | - |
| `segments` | 每个单词的文本 | - |
| `separators` / `separator_token_ids` | 每个 segment 的分隔符字符 / id | - |

### 3.3 对齐层面的结论

1. **CTC 只用 `ctc_token_ids`。** `forced_align_ctc_logits` 针对不含分隔符的目标序列做对齐。
   因此标点**不占用任何编码器帧**，也**不消耗任何语音 token**。
2. **标点没有时间区间。** `align_text_tokens_to_speech_tokens` 只把语音 token 分配给文本 token（音素）。
   因此单词之间在时间上直接相邻：前一个单词的结束即后一个单词的开始。
3. **标点仍存在于序列中。** 它作为分隔符 token 出现在 `shifted_token_ids_with_separators`、
   交错序列和块序列里，始终位于词与词之间（从不与语音绑定）。
4. **句末标点是结构性的。** `.`、`!`、`?` 在构造块时充当决策点（见第 6 节）；
   `,` 与空格属于“继续型”分隔符。
5. **在 `units` 输出中**（`format_units_text`），每个 segment 的音素符号按序输出，
   紧随其后追加该 segment 的分隔符，最后用空格连接 —— 也就是说标点在文本视图中被保留，
   尽管它在时间上长度为 0。

## 4. CTC 路径后处理

- `forced_align_ctc_logits` 调用 `torchaudio.functional.forced_align`，返回逐帧的原始路径
  （长度 = 编码器帧数 `T`）与逐帧分数，`blank_id = 0`。
- `postprocess_alignment_path` 产生：
  - `raw_path`：原始路径（保留 blank）。
  - `filled_path`：每个 blank 替换为其后第一个非 blank；结尾的 blank 替换为最后一个非 blank。
  - `token_path` + `token_lengths`：在文本元素层面折叠。元素前的 blank 计入该元素；结尾的 blank 加到最后一个元素；
    **被 blank 隔开的两个相同标签仍算作两个独立元素**。
- 一致性校验：`token_path` 必须等于 `ctc_token_ids`，否则抛出 `ValueError`。

## 5. 文本 token 到语音 token 区间的映射

`align_text_tokens_to_speech_tokens` 按顺序遍历文本 token：

```
start = previous_end
cumulative_length += token_length[i]
end   = len(speech_tokens)              # 最后一个 token 吸收整个尾部
      = floor(ratio * cumulative_length) # 其余情况
start, end = 裁剪到 [0, len(speech_tokens)]
```

- 本配置下 `ratio = 3.0`，即一个编码器帧的时长对应三个语音 token。
- 最后一个文本 token 会吸收**所有剩余的尾部语音 token**，从而保证完整覆盖。
- 函数会断言各区间连续、起点为 `0`、终点为 `len(speech_tokens)`（无空隙、无重叠、无剩余语音）。

由于区间边界落在语音 token 网格上（75 token/秒 = 13.33 ms），即使 CTC 对齐本身是在
40 ms 的编码器网格上计算，音素边界实际上也会被量化到 13.33 ms。

## 6. 块构造（`build_unit_speech_blocks`）

块粒度 = **当前单词**（`data.block_type = "biword"`）。

每个块的结构为：

```
当前词 units + 它的分隔符 + [可选的下一词 units] + 当前词语音 token + eob
```

算法（对每个文本 segment / 单词）：

1. 用当前词的 unit（音素）token 开始这个块。
2. 若有分隔符，则追加该词的分隔符 token。
3. 追加“未来文本”部分：
   - **整句的最后一个词** -> 追加 `eos`；
   - 否则若分隔符是句末（`. ! ?`）-> **不追加任何东西**
     （既不加下一词，也不加 `eos`）；
   - 否则（空格或 `,`）-> 追加**下一词的 units**（向前看一个词）。
4. 追加当前词的语音 token（由其音素 token 聚合而来）。
5. 追加 `eob`。

说明：

- 标点属于块的文本前缀，但从不携带语音 token。
- 非末尾的词若没有分隔符，会抛出
  `"Missing mandatory separator between adjacent text segments"`。
- 由于普通块包含“当前词 + 下一词”，该模式被命名为 **"biword"**。

展平：`flatten_unit_speech_blocks` 按顺序把所有块拼成一维的 `seqs_sequence`，
`align_data.py` 将其存为 `seqs` 字段（`" ".join(map(str, result.seqs_sequence))`）。

对齐侧的构造函数与推理侧的 `generate_unit_list`（`src/utils.py`）实现的是同一套
“当前词 + 分隔符 + 可选的下一词”布局；`extract_unit_from_seqs`（`src/utils.py`）在推理时
从每个块中恢复出 `当前词 units + 分隔符` 前缀。

## 7. 交错序列（逐 token 视图）

`build_interleaved_sequence` 对每个 segment，依次输出每个文本 token 的偏移 id 及其对齐到的语音 token；
在每个 segment 之后插入分隔符 token。这是一个用于检查的逐 token 视图，与块/`seqs` 训练序列不同。
标点在此处会被插入，但仍不与语音绑定。

## 8. TextGrid 输出

`build_textgrid` 渲染 Praat TextGrid（long 格式），包含两个 `IntervalTier`：

- `phones`：每个文本 token 一个区间。`xmin = speech_start / 75`，`xmax = speech_end / 75`，
  text = `vocab[ctc_token_id]`（音素符号）。
- `words`：每个 segment 一个区间。起点 = 首个 token 的 `speech_start`，终点 = 末个 token 的
  `speech_end`，text = 该 segment 字符串。

时间轴：`秒 = speech_token_index / config["nac"]["tokens_per_second"]`（75）。

`xmax` = `max(audio_duration, aligned_duration)`，其中 `aligned_duration = last.speech_end / 75`；
当真实音频更长时，会在**两个** tier 末尾各追加一个空区间，使它们继续覆盖 `[xmin, xmax]`。

标点永远不会形成自己的区间。由于分隔符不携带语音 token，相邻的单词区间直接相连，
`words` tier 在 `.`/`,`/空格出现的位置不会留有间隙。

## 9. 示例

文本 `"ab c."`，编码为 `"a"->4097, "b"->4098, "c"->4099`（偏移 4096），
分隔符 `" "=4196`、`"."=4198`，`eob=4201`，`eos=4203`：

- `segments = ["ab", "c"]`，`separators = [" ", "."]`，`segment_lengths = [2, 1]`。
- 块 0（词 `"ab"`，分隔符为空格，存在下一词 `"c"` 且非句末）：
  `[4097, 4098, 4196, 4099, <“ab”的语音>, 4201]`
- 块 1（词 `"c"`，分隔符 `"."`，整句最后一个词）：
  `[4099, 4198, 4203, <“c”的语音>, 4201]`

`seqs` = 块 0 拼上块 1（展平）。
在 TextGrid 中，`words` 恰好有两个区间（`ab`、`c`）且彼此相接，`phones` 每个音素一个区间；
两个 tier 都不包含 `.` 的区间。

## 10. 注意事项

- `separators` 可能比 `segments` 少（最后一个词没有尾随标点）或多（重复/相邻分隔符，例如两个空格）。
  只有下标 `< len(separators)` 的 segment 会获得分隔符 token，且“缺少必需分隔符”的报错只会在
  非末尾词缺分隔符时触发。
- 对齐侧的块构造函数会给最后一个词的块追加 `eos`，而推理侧的 `generate_unit_list`
  （`src/utils.py`）当前把那处 `eos` 追加注释掉了（最后一个词不加 `eos`）。
  在依赖该行为前，请确认这一不对称是否为有意设计。
- 标点只影响块的边界与文本前缀，从不消耗语音 token 或时间。

## 11. 函数索引

| 函数 | 作用 |
| --- | --- |
| `_get_separator_mapping` | 读取 `data.separators` |
| `_get_text_offset` | 文本 token 偏移（= `nac.codebook_size`） |
| `_get_ratio` | `nac.tps / encoder.tps` |
| `split_text_by_config_separators` | 把文本切成 segments + separators |
| `encode_text_with_separators` | 构造三条 token 序列 |
| `forced_align_ctc_logits` | CTC 强制对齐 |
| `replace_blanks_in_path` / `collapse_path_to_tokens` / `postprocess_alignment_path` | 路径后处理 |
| `align_text_tokens_to_speech_tokens` | 文本 token -> 语音区间映射 |
| `build_interleaved_sequence` | 逐 token 交错序列 |
| `build_unit_speech_blocks` / `flatten_unit_speech_blocks` | 块构造与展平 |
| `build_textgrid` | Praat TextGrid 渲染 |
| `align_text_logits_and_speech_tokens` | 顶层入口 |
