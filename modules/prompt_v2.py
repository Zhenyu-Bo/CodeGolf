# Prompt templates for the new agent
PROBLEM_UNDERSTANDING_PROMPT = """You are an expert Python programmer specializing in code golf and abstract reasoning puzzles. 

## Task: Problem Understanding and Analysis

I will provide you with a grid transformation task. Your goal is to understand the problem thoroughly and identify the transformation rule.

## Problem Context:

1. Input-output examples:
{examples_str}

2. Generator code:
```python
{generator_code}
```

3. Current working solution:
```python
{current_solution}
```

## Your Task:

1. **Analyze the transformation rule**: Study the input-output examples to understand what transformation is being applied.

2. **Understand the generator constraints**: Examine the generator code to identify any implicit constraints or patterns that could simplify the solution.

3. **Identify optimization opportunities**: Look at the current solution and identify where it might be doing unnecessary work or could be simplified based on the generator constraints.

## Output Format:

Please provide your analysis in the following format:

**Problem Understanding:**
[Describe what the task is asking for in clear terms]

**Transformation Rule:**
[Explain the rule that transforms input to output]

**Generator Insights:**
[List any constraints or patterns from the generator that could simplify the solution]

**Optimization Opportunities:**
[Identify potential areas where the current solution could be simplified or shortened]

**Confidence Level:**
[Rate your confidence in understanding the problem on a scale of 1-10]

Take your time to analyze thoroughly. The better you understand the problem, the better we can optimize the solution.
"""

TRICK_APPLICATION_PROMPT = """You are an expert in Python code golf optimization. I will provide you with a working solution and a set of optimization tricks to apply.

## Current Solution:
```python
{current_code}
```

## Available Tricks:
{tricks_info}

## Your Task:

For each trick provided above, try to apply it to the current solution. For each trick:

1. Determine if the trick is applicable to the current code
2. If applicable, show the modified code after applying the trick
3. Verify that the modified code is shorter and still logically correct
4. If not applicable, explain why

## Output Format:

For each trick, provide:

**Trick {{i}}: {{trick_description}}**
- **Applicable:** [Yes/No]
- **Modified Code:** 
```python
[modified code if applicable, or "N/A" if not applicable]
```
- **Length Change:** [original_length → new_length (change: +/-X)]
- **Explanation:** [Brief explanation of the change or why it's not applicable]

## Important Notes:
- Only apply tricks that actually make the code shorter
- Ensure the modified code maintains the same functionality
- Be conservative - if you're not sure a trick applies, mark it as "No"
- Focus on meaningful optimizations, not micro-optimizations that might break functionality
"""

SELF_GOLF_PROMPT = """You are a Python code golf expert. Your task is to make the following code as short as possible while maintaining its functionality.

## Current Solution:
```python
{current_code}
```

## Generator Code (for context):
```python
{generator_code}
```

## Your Task:

Look for major algorithmic improvements and simplifications. Focus on:

1. **Algorithmic changes**: Can you solve the problem with a fundamentally different approach?
2. **Generator insights**: Are there constraints in the generator that allow you to simplify assumptions?
3. **Data structure optimizations**: Can you use more efficient data structures or operations?
4. **Logic simplification**: Can you combine or eliminate steps in the current logic?

## Guidelines:

- Prioritize substantial changes over minor syntactic tricks
- Consider the generator code carefully - it often reveals constraints that can simplify the solution
- Don't just apply micro-optimizations; look for structural improvements
- Ensure your solution maintains correctness

## Output Format:

**Proposed Optimization:**
```python
[your optimized code]
```

**Key Changes:**
[List the main changes you made and why they work]

**Length Improvement:**
[Original: X chars → New: Y chars (saved: Z chars)]

**Reasoning:**
[Explain your optimization strategy and how you ensured correctness]
"""

KNOWLEDGE_SCAN_PROMPT = """You are a Python code golf expert with extensive knowledge of optimization techniques. 

## Current Solution:
```python
{current_code}
```

## Task Context:
{problem_context}

## Your Task:

Scan your knowledge base for applicable code golf tricks and techniques. Look for:

1. **Python built-in optimizations**: More efficient built-in functions or methods
2. **Syntactic shortcuts**: Shorter ways to express the same logic
3. **Mathematical tricks**: Clever mathematical approaches
4. **Data structure tricks**: More efficient ways to handle data
5. **Control flow optimizations**: Shorter conditional logic or loops

## Output Format:

**Found Techniques:**

**Technique 1: [Name]**
- **Description:** [What this technique does]
- **Application:** 
```python
[how to apply it to current code]
```
- **Savings:** [estimated character savings]

**Technique 2: [Name]**
[continue for each technique found...]

**Final Optimized Code:**
```python
[your best optimized version combining applicable techniques]
```

**Total Improvement:**
[Original: X chars → Final: Y chars (saved: Z chars)]

Focus on techniques that provide meaningful character savings and maintain code correctness.
"""
