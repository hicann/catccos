# Pre-commit 代码检查配置说明

本文档说明本次为 catccos 项目的 pre-commit 代码检查配置。

## 新增检查清单

### 1. 通用基础检查 ([pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks))

| Hook | 功能 |
|------|------|
| `trailing-whitespace` | 删除行尾空格 |
| `end-of-file-fixer` | 确保文件以空行结尾 |
| `check-yaml` | 验证 YAML 语法 |
| `check-json` | 验证 JSON 语法 |
| `check-added-large-files` | 阻止提交大文件（默认 >500KB）|
| `check-merge-conflict` | 检测合并冲突标记 |
| `detect-private-key` | 检测私钥/敏感信息泄露 |

### 2. Python 代码检查 ([ruff](https://github.com/astral-sh/ruff))

- **ruff lint**: 代码规范检查（规则配置见 `pre-commit/pyproject.toml`）
  - 启用 `D209`（多行文档字符串格式）
  - 启用 `SIM115`（推荐 `with` 语句）
- **ruff format**: 自动代码格式化
- **配置**: 使用 `pre-commit/pyproject.toml`，兼容 Python 3.10+

### 3. Python 安全扫描 ([bandit](https://github.com/PyCQA/bandit))

- 扫描常见的 Python 安全漏洞（如硬编码密码、不安全的 eval、SQL 注入等）
- **配置**: 中危及以上级别，配置见 `pre-commit/pyproject.toml`

### 4. 拼写检查 ([typos](https://github.com/crate-ci/typos))

- 检测代码和文档中的拼写错误
- **配置**: `pre-commit/typos.toml`，已添加昇腾生态相关词汇白名单

### 5. C++ 代码格式化 ([clang-format](https://clang.llvm.org/docs/ClangFormat.html))

- 自动格式化 C/C++ 代码
- **风格**: Google Style，4 空格缩进，120 列限制，Allman 大括号风格
- **匹配文件**: `*.c, *.h, *.cpp, *.hpp, *.cc, *.hh, *.cxx, *.hxx`

## 配置文件说明

| 文件 | 用途 |
|------|------|
| `.pre-commit-config.yaml` | pre-commit 主配置，定义 hooks |
| `pre-commit/pyproject.toml` | Python 工具配置（ruff、bandit）|
| `pre-commit/typos.toml` | 拼写检查词典白名单 |

## 使用方式

```bash
# 安装 hook（首次）
pre-commit install

# 对所有文件运行检查
pre-commit run --all-files

# 提交时自动触发检查
git commit -m "..."
```
