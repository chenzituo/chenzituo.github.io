---
title: "A first note"
date: 2026-09-29T09:00:00-04:00
draft: false
tags: [notes]
summary: "A sample post showing the format for future notes: clear sections, equations, code, and references."
author: "Zituo · sample post"
math: true
---
This is a sample post to demonstrate the blog’s reading layout. Replace it with your first note when you’re ready.

## Start with a question

A useful note can begin with a single question. State it plainly, explain why it matters, and work toward an answer one step at a time.

## Work through an example

Equations can sit alongside the explanation. For a sequence of observations \(x_1, \ldots, x_n\), their arithmetic mean is

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
$$

A small code example makes the calculation concrete:

```python
def mean(values):
    if not values:
        raise ValueError("At least one observation is required")
    return sum(values) / len(values)
```

## Leave a trail

Link to original sources, explain assumptions, and separate observations from open questions. The table of contents and heading links help readers return to a specific part of a longer post.

> A note does not have to settle a topic. It can record what is understood so far.

## References

1. [Hugo documentation](https://gohugo.io/documentation/), for writing and publishing with this site.
2. [PaperMod](https://github.com/adityatelange/hugo-PaperMod), the theme used for the reading layout.
