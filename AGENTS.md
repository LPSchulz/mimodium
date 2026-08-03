# Documentation guidance

- Keep type docstrings as short as possible while remaining expressive.
- Give task docstrings enough detail to explain not only what the task does and
  how to use it, but also why it exists. Include examples for complicated
  algorithms when useful.
- Use `:type:` links frequently in type documentation. They are generally
  unnecessary in task documentation because the task's `__call__` method
  already provides those links.
- When a task accepts configurable input types, always explain how to use them
  with `:code:` markup.
