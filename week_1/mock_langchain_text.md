# Week 1 – Introduction to Node Parsing

This document is a mock file created for demonstrating how
LangchainNodeParser works together with RecursiveCharacterTextSplitter.

The goal of this example is to show how text-based splitting differs
from syntax-aware code splitting.


## 1. What is a Node?

In LlamaIndex, a **Node** represents a chunk of text that is used
as the basic unit for indexing and retrieval.

Each node typically contains:
- A portion of the original text
- Metadata such as source file name
- Optional relationships to other nodes

Nodes are created by applying a *Node Parser* to documents.


## 2. Text-Based Splitting

Text-based splitting focuses on:
- Character count
- Line breaks
- Paragraph boundaries

It does **not** understand programming language syntax.

This makes it suitable for:
- Markdown documents
- Plain text notes
- Documentation
- Blog posts


## 3. Example: Recursive Character Splitting

RecursiveCharacterTextSplitter attempts to split text using
a hierarchy of separators, such as:

1. Double newlines
2. Single newlines
3. Spaces
4. Characters

This approach helps preserve semantic structure where possible,
while still enforcing chunk size limits.


### Example Code Block (treated as plain text)

```python
def hello_world():
    print("Hello, world!")

for i in range(3):
    hello_world()
```

Note that the splitter does not understand Python syntax.
It simply treats this as text.

## 4. When to Use LangchainNodeParser

You should consider using LangchainNodeParser when:

- Your data is mostly natural language
- You want predictable chunk sizes
- You do not need syntax-aware parsing

For code-heavy use cases, a CodeSplitter is usually more appropriate.

## 5. Summary

In this mock document, we intentionally mix:

- Headings
- Paragraphs
- Lists
- Code blocks

This allows you to clearly observe how
RecursiveCharacterTextSplitter creates nodes
based purely on textual structure