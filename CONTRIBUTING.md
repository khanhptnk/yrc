# Contributing to YRC

Welcome! 👋

We're excited you're interested in contributing to **YRC (Yield and Request Control)** — a research framework for studying decision delegation in human-AI collaboration. Contributions of all kinds are welcome: new features, bug fixes, documentation improvements, tests, or ideas for future work.

This document outlines our contribution process and standards.

---

## 🧠 Before You Start

- **Familiarize yourself** with the YRC codebase and documentation:  
  [https://yrc.readthedocs.io](https://yrc.readthedocs.io)

- **Check open issues** or create a new one to propose your idea or report a bug.  
  We recommend discussing major changes with us before investing significant effort.

---

## 🚀 How to Contribute

### 1. Fork the Repository

Click "Fork" on [GitHub](https://github.com/khanhptnk/yrc), then clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/yrc.git
cd yrc
```

### 2. Set Up the Environment

Install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This includes tools like `pytest`, `ruff`, and documentation support.

---

### 3. Make Your Changes

Please follow these guidelines:

- Keep changes focused on a single issue or feature.
- Write clear, well-documented code using **type hints** and **NumPy-style docstrings**.
- Add or update **tests** under the `tests/` directory.
- Add or update **documentation** if needed (under `docs/`).
- Ensure your code passes formatting and lint checks.

---

### 4. Code Formatting and Linting

We use [`ruff`](https://docs.astral.sh/ruff/) for both **formatting** and **linting**.

Before committing, run:

```bash
ruff format .       # Auto-format your code
ruff check .        # Lint for style and logic errors
```

To automatically fix most lint errors:

```bash
ruff check . --fix
```

---

### 5. Run Tests

Make sure the test suite passes:

```bash
pytest
```

Include tests for any new functionality or bug fixes.

---

### 6. Commit and Push

```bash
git checkout -b your-feature-name
git commit -m "Add explanation of policy delegation logic"
git push origin your-feature-name
```

Use clear and descriptive commit messages.

---

### 7. Open a Pull Request

On GitHub, open a Pull Request (PR) against the `main` branch.

Please include:

- A summary of your changes
- Related issue numbers (e.g., `Closes #23`)
- Notes on testing or design decisions

We’ll review and provide feedback as soon as possible!

---

## 🧾 Contributor Agreement

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).  
See [LICENSE](LICENSE) for more details.

---

## 🙏 Acknowledgment

We deeply appreciate all contributions — large and small.  
Your involvement helps make YRC more robust, usable, and impactful for the research community.

---

## 📬 Questions?

Open an issue or contact the maintainers directly at [khanhptnk@gmail.com].

Thanks again for contributing to YRC!

