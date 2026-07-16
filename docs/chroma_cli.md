## Chroma 本地命令行查看

前提：

- 确保已安装依赖：`pip install -r requirements.txt`
- 确保 `.env` 里配置了 `CHROMA_PATH` / `CHROMA_COLLECTION`（示例见 [.env.example](../.env.example)）

### 方案 A：一条命令直接看

列出所有 collections：

```bash
python -c "from service.chroma_store import get_chroma_manager; print(get_chroma_manager().list_collection_names())"
```

查看默认 collection（`CHROMA_COLLECTION`，默认 `knowledge_chunks`）的数量 + 预览前 5 条：

```bash
python -c "from service.chroma_store import get_chroma_manager; print(get_chroma_manager().preview_collection(limit=5))"
```

指定 collection 名查看：

```bash
python -c "from service.chroma_store import get_chroma_manager; print(get_chroma_manager().preview_collection('knowledge_chunks', 5))"
```

### 方案 B：用脚本查看（更推荐）

列出 collections：

```bash
python chroma_inspect.py list
```

查看 collection 统计信息（默认用 `CHROMA_COLLECTION`）：

```bash
python chroma_inspect.py info
```

预览前 N 条数据（默认 5 条）：

```bash
python chroma_inspect.py preview --limit 5
```

指定 collection：

```bash
python chroma_inspect.py preview --collection knowledge_chunks --limit 10
```

### 说明

如果看到 `Failed to send telemetry event ...`，一般不影响本地读写与查询结果。
