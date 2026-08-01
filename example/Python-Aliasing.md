+++
name = "Python aliasing"
+++

G: In Python 3, `ys = xs` aliases a list; `ys = xs[:]` copies it; `append` mutates the referenced list.
Q: What does this Python 3 code print?

```python
{{choose distinct digits a, b, and c from 1 through 9, then emit exactly four lines with them substituted: `xs = [a, b]`; either `ys = xs` or `ys = xs[:]`; `ys.append(c)`; `print(xs, ys, ys is xs)`; code only}}
```

A: {{give the exact single stdout line produced by Q, including Python list spacing and Boolean capitalization; no explanation}}
