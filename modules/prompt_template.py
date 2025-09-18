"""
Prompt templates for Golf Agent
"""

# Variant generation prompt template
VARIANT_GENERATION_PROMPT = """You are an expert Python programmer specializing in code golf and abstract reasoning puzzles. You need to understand and finish the following task:

## Task Description:
You are given a grid transformation task where you need to generate multiple fundamentally different algorithmic variants to solve the same problem.
I will provide you with:
1. The input-output examples of the task. The input is a grid represented as a list of lists of integers, and the output is the transformed grid following a hidden rule.
2. The code that generates these examples (the generator)
3. The current solution code named `p` which has been proved correct

Your task is to analyze the given problem and generate {n_variants} DIFFERENT algorithmic approaches to solve it.

## Problem Context
Here are the provided informations about the task:

1. Input-output examples:
{examples_str}

2. Generator code:
```python
{generator_code}
```

3. Current solution code:
```python
{initial_solution}
```

## Analysis and Understanding Instructions
Here are some instructions to help you analyze and generate the variants:

**Problem Analysis Instructions:**
1. Analyze the input-output examples, the generator code, and the current solution code to understand the transformation rule.
2. Analyze the generator code again to identify any implicit constraints or conditions that can help simplify the solution.
3. Understand the core logic and transformation rule implemented in the current solution. Then try to identify fundamentally different approaches to achieve the same result.

**Note:**
1. the generator code is very important to the task because it may contain implicit constraints and conditions that you can use to reduce some unnecessary processing in the provided code so that shorten it.
For example:
    a. If the generator only creates grids with certain properties (e.g., specific value ranges, patterns), you can leverage these properties to simplify your variants.
    b. If the generator generates different connected components with unique color, you can directly collect all pixel positions by color instead of using traditional algorithms for finding connected components such as DFS or BFS.
2. The logic and transformation rule implemented in the current solution are correct, but there may be more concise logic or rules. Even if the rule is one and only, there may be multiple ways to express it more concisely. So you need to analyze and understand the input-output examples and generator code carefully to identify any potential simplifications or optimizations.

## Requirements
Here are the requirements and guidelines for generating the variants:

1. **Algorithmic Diversity:** Focus on fundamentally different strategies (e.g., iteration pattern, data structures, recursive vs. iterative) or different algorithms. Avoid trivial changes.
2. **Correctness:** All variants must pass the examples and adhere to the rule defined by the generator code. Test your logic mentally against the examples before providing code.
3. **Length:** Try to using more concise strategies or data structures and generate shorter code, but do not sacrifice clarity or correctness.
4. **Completeness:** Each variant must be a complete, runnable function.
5. **Entry Point:** The solution code must define a callable object named `p` (e.g., a function or a lambda). This callable must accept one argument (the input grid) and return the corresponding output grid. Note: Your code must include function `p`, but it does not have to be the only function or class in your code, which means you can define helper functions or classes if needed.
6. **Libraries:** You are **only allowed to use standard Python libraries**. No third-party libraries like `numpy`, `scipy`, etc., are permitted.
7. **Output Format:** Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
<answer_begin>
### Variant 1: [Descriptive Strategy Name]
**Core Strategy:** [Explain the main algorithmic approach in 1-2 clear sentences]
**Key Differences:** [Specifically how this differs from the original solution's approach]
**Implementation Notes:** [Any important technical details about this approach]

```python
def p(g):
    # Your complete implementation here
    # Include brief comments explaining key steps
    # Ensure this is syntactically correct Python
    pass
```

### Variant 2: [Descriptive Strategy Name]
**Core Strategy:** [Explain the main algorithmic approach in 1-2 clear sentences]
**Key Differences:** [Specifically how this differs from the original solution's approach]
**Implementation Notes:** [Any important technical details about this approach]

```python
def p(g):
    # Your complete implementation here
    pass
```

(Continue for all {n_variants} variants...)
</answer_end>

Now follow the above instructions and requirements, begin your analysis and variant generation, finally provide {n_variants} different variants in the required format.
"""

# Optimization prompt template
OPTIMIZATION_PROMPT = """You are an expert Python programmer specializing in code golf and abstract reasoning puzzles. You need to understand and finish the following task:

## Task Description:
You are given a grid transformation task where you need to generate multiple fundamentally different algorithmic variants to solve the same problem.
I will provide you with:
1. The input-output examples of the task. The input is a grid represented as a list of lists of integers, and the output is the transformed grid following a hidden rule.
2. The code that generates these examples (the generator)
3. The current solution code named `p` which has been proved correct
4. The history of previous optimization attempts

Your task is to analyze the given problem and optimize the current solution code `p` to make it as short as possible while maintaining its exact functionality and correctness.
You had better focus on global optimizations rather than local ones, such as changing the algorithm or data structures used, rather than just tweaking syntax.

## Problem Context

### Task Examples:
Here are the input-output examples of the task:
{examples_str}

### Generator Code:
Here is the code that generates the input-output examples. Analyze it carefully to understand the underlying transformation rules:
```python
{generator_code}
```

### Current Code to shorten:
Here is the current solution code that needs shortening:
```python
{code}
```

### History of Previous Shortening Attempts:
Here is the history of previous shortening attempts, if any. You can analyze these attempts to identify what has already been tried and what worked or didn't work, and use this information to guide your shortening process:

{history_str}

## Analysis and Understanding Instructions
Here are some instructions to help you analyze and optimize the code:
**Problem Analysis Instructions:**
1. Analyze the input-output examples, the generator code, and the current solution code to understand the transformation rule.
2. Analyze the generator code again to identify any implicit constraints or conditions that can help simplify the solution.
3. Understand the core logic and transformation rule implemented in the current solution. Then try to identify areas where the code can be shortened without changing its functionality.
4. Review the history of previous shortening attempts to avoid repeating ineffective strategies and to build upon successful ones.
5. Focus on both algorithmic and syntactic optimizations to achieve the shortest possible code.

**Note:**
1. the generator code is very important to the task because it may contain implicit constraints and conditions that you can use to reduce some unnecessary processing in the provided code so that shorten it.
For example:
    a. If the generator only creates grids with certain properties (e.g., specific value ranges, patterns), you can leverage these properties to simplify your variants.
    b. If the generator generates different connected components with unique color, you can directly collect all pixel positions by color instead of using traditional algorithms for finding connected components such as DFS or BFS.
2. The logic and transformation rule implemented in the current solution are correct, but there may be more concise logic or rules. Even if the rule is one and only, there may be multiple ways to express it more concisely. So you need to analyze and understand the input-output examples and generator code carefully to identify any potential simplifications or optimizations.

## Requirements
Here are the requirements and guidelines for shortening the code:

1. **Entry Point:** The solution code must define a callable object named `p` (e.g., a function or a lambda). This callable must accept one argument (the input grid) and return the corresponding output grid. Note: Your code must include function `p`, but it does not have to be the only function or class in your code, which means you can define helper functions or classes if needed.
2. **Libraries:** You are **only allowed to use standard Python libraries**. No third-party libraries like `numpy`, `scipy`, etc., are permitted.
3. **Correctness:** The shortened code must produce the same output as the original code for all valid inputs defined by the generator. You must mentally verify the correctness of your shortened code against the provided examples.
4. **Length:** Focus on achieving the shortest possible code while maintaining clarity and correctness. Avoid unnecessary complexity that does not contribute to shortening.
5. **Big Change Encouragement:** You are encouraged to make significant changes to the algorithm or data structures used if it leads to a shorter implementation, rather than just making minor syntactic tweaks.
6. **Output Format:** Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
<answer_begin>
My Strategy: [expplain your optimization strategy briefly]

My optimized code:
```python
def p(g):
    # Your optimized implementation
    # Focus on shortest byte count while preserving functionality
    pass
```
</answer_end>

Now, follow the above instructions and requirements, begin your optimization, finally provided your answer in the required format({shortest_hint}).
"""

# Tricks application prompt template
TRICKS_PROMPT = """Apply specific code golf tricks to make this code shorter while maintaining correctness.

## Current Code:
```python
{code}
```

## Available Golf Tricks:
{tricks_str}

## Instructions:
1. Scan the tricks list and your own knowledge base, analyze which tricks can be safely applied to the current code and shorten the current code indeed
2. Apply applicable tricks step by step
3. Ensure the function still works correctly
4. Provide the final optimized code and explain which tricks were applied

## Requirements:
1. **Entry Point:** The solution code must define a callable object named `p` (e.g., a function or a lambda). This callable must accept one argument (the input grid) and return the corresponding output grid. Note: Your code must include function `p`, but it does not have to be the only function or class in your code, which means you can define helper functions or classes if needed.
2. **Libraries:** You are **only allowed to use standard Python libraries**. No third-party libraries like `numpy`, `scipy`, etc., are permitted.
3. **Correctness:** The optimized code must produce the same output as the original code for all valid inputs defined by the generator. You must mentally verify the correctness of your optimized code against the provided examples.
4. **Length:** Focus on achieving the shortest possible code while maintaining clarity and correctness. Avoid unnecessary complexity that does not contribute to shortening.

## Output Format:
Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
<answer_begin>
Applied Tricks: [List the tricks you applied]
Optimized Code:
```python
def p(g):
    # Your optimized implementation
    # Focus on shortest byte count while preserving functionality
    pass
```
</answer_end>

Now follow the above instructions and requirements, begin your optimization, finally provide your answer in the required format({shortest_hint}).
"""

# Failed variant fixing prompt template
FIX_FAILED_VARIANT_PROMPT = """You are an expert Python debugger and problem solver specializing in code golf.

A code variant has failed and needs to be fixed while preserving its core algorithmic approach.

## Problem Context

### Error Information:
**Error:** {error_summary}

### Failed Code:
```python
{original_code}
```

### Original Strategy: 
{strategy}

## Requirements
1. **Entry Point:** The solution code must define a callable object named `p` (e.g., a function or a lambda). This callable must accept one argument (the input grid) and return the corresponding output grid. Note: Your code must include function `p`, but it does not have to be the only function or class in your code, which means you can define helper functions or classes if needed.
2. **Libraries:** You are **only allowed to use standard Python libraries**. No third-party libraries like `numpy`, `scipy`, etc., are permitted.
3. **Correctness:** The fixed code must pass the provided examples and adhere to the rule defined by the generator code. You must mentally verify the correctness of your fixed code against the provided examples.
4. **Algorithm Preservation:** If the original algorithmic approach is correct, preserve it as much as possible. If not, there are no restrictions on changing the algorithm.
5. **Output Requirements:** Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
<answer_begin>
Problem Analysis: [Briefly analyze the error and identify the root cause]
Applied Fixes: [List the specific fixes you applied]
Fixed Code:
```python
def p(g):
    # Your fixed implementation
    # Ensure it works correctly and preserves the original approach
    pass
```
</answer_end>

Follow the above instructions and requirements, begin your debugging and fixing now.
"""

# Knowledge base tricks scanning prompt template
KNOWLEDGE_BASE_TRICKS_PROMPT = """You are an expert Python code golf specialist with deep knowledge of optimization techniques.

Your task is to analyze the given code and apply proven code golf optimizations from your knowledge base.

## Code Analysis Target

### Current Code:
```python
{code}
```

Instructions:
1. Analyze the current code structure and identify patterns that can be optimized.
2. Scan your knowledge base of code golf tricks and techniques, identifying those that can be applied to the current code.
3. If you can not find any applicable by yourself, consider the following tricks:
    - Structural optimizations:
        - **String manipulation tricks:** Slicing, indexing, formatting shortcuts
        - **Lambda optimizations:** For simple function definitions
        - **Recursive patterns:** When they reduce code vs iterative approaches
        - **Mathematical formulas:** Replace algorithmic computation when possible
    - Loop and iteration optimizations: 
        - Use list comprehensions or generator expressions instead of loops where possible.
        - Use `enumerate()` instead of manual indexing.
        - Use `zip()` for parallel iteration over multiple sequences.
        - Use `all()` or `any()` for boolean accumulation instead of loops.
        - Flatten nested loops into a single loop with mathematical calculations when possible.
    - Expression and operator optimizations:
        - Use the walrus operator (`:=`) to combine assignment and condition checks.
        - Use chained comparisons (e.g., `a < b < c`) instead of multiple `and` conditions.
        - Simplify boolean expressions using boolean algebra (e.g., `not(not x)` to `bool(x)`).
        - Use tuple indexing for conditional assignments (e.g., `(b, a)[cond]`).
        - Use mathematical shortcuts (e.g., `x // 1` for `int(x)`, `x ** 2` instead of `x * x`).
        - Use bitwise operations when applicable and shorter.
        - Use built-in functions like `sum()`, `max()`, `min()` instead of manual loops for aggregation.
        - Use `map()` and `filter()` for sequence transformations.
        - Use string methods like `str.join()` and slicing tricks for string manipulations.
        - Use set operations for unique elements and intersections.
        - Use `sorted()` with key functions for sorting.
        - Use tuple unpacking for multiple assignments (e.g., `a, b, c = sequence`).
        - Use `dict.get()` or `setdefault()` for default dictionary patterns.
    - Control flow optimizations:
        - Short-circuit evaluation with `and`/`or`.
        - Remove early returns(need to guarantee correctness).
4. Apply the identified tricks step by step, ensuring that the code remains correct and functional.

### Priority Order:
1. **High-impact algorithmic changes** (biggest byte savings)
2. **Variable and expression optimization** (medium impact)
3. **Built-in function substitutions** (reliable savings)
4. **Syntax and spacing optimization** (final polish)

### Safety Guidelines:
- **Preserve exact functionality** - no behavior changes
- **Maintain edge case handling** - don't break boundary conditions
- **Keep readability sufficient** for verification
- **Test each optimization** mentally against examples

## Requirements

1. **Entry Point:** The solution code must define a callable object named `p` (e.g., a function or a lambda). This callable must accept one argument (the input grid) and return the corresponding output grid. Note: Your code must include function `p`, but it does not have to be the only function or class in your code, which means you can define helper functions or classes if needed.
2. **Libraries:** You are **only allowed to use standard Python libraries**. No third-party libraries like `numpy`, `scipy`, etc., are permitted.
3. **Correctness:** The optimized code must produce the same output as the original code for all valid inputs defined by the generator. You must mentally verify the correctness of your optimized code against the provided examples.
4. **Length:** Focus on achieving the shortest possible code while maintaining clarity and correctness. Avoid unnecessary complexity that does not contribute to shortening.
5. **Output Requirements:** Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
<answer_begin>
Applied Tricks: [List the tricks you applied]
```python
def p(g):
    # Your fully optimized code
    # Apply maximum safe optimizations
    pass
```
</answer_end>

If NO tricks can safely improve this code, respond with exactly:
<answer_begin>
NO_APPLICABLE_TRICKS
</answer_end>

Now follow the above instructions, find and apply all the useful tricks and provide your optimized code({shortest_hint}).
"""

# Provided tricks application prompt template
PROVIDED_TRICKS_PROMPT = """Apply specific code golf tricks to make this code shorter while maintaining correctness.

## Current Code:
```python
{code}
```

## Available Golf Tricks Database:
{tricks_str}

## Application Process:
1. **Analyze current code structure** - identify patterns that match the tricks
2. **Select applicable tricks** - only choose tricks that fit the current code
3. **Apply tricks incrementally** - one trick at a time to avoid errors
4. **Verify byte count reduction** - ensure each trick actually makes code shorter
5. **Test correctness** - ensure functionality is preserved

## Application Guidelines:
- **Prioritize high-impact tricks** that save the most bytes
- **Combine compatible tricks** when possible
- **Be conservative** - only apply tricks you're confident about
- **Maintain readability** enough to verify correctness
- **Avoid over-optimization** that might introduce bugs

## Safety Checks:
- Does the trick actually reduce byte count in this specific context?
- Does it preserve the exact same functionality and edge case behavior?
- Are there any Python version compatibility issues?
- Does it maintain the correct function signature `def p(g):`?

## Requirements:
1. **Entry Point:** The solution code must define a callable object named `p` (e.g., a function or a lambda). This callable must accept one argument (the input grid) and return the corresponding output grid. Note: Your code must include function `p`, but it does not have to be the only function or class in your code, which means you can define helper functions or classes if needed.
2. **Libraries:** You are **only allowed to use standard Python libraries**. No third-party libraries like `numpy`, `scipy`, etc., are permitted.
3. **Correctness:** The optimized code must produce the same output as the original code for all valid inputs defined by the generator. You must mentally verify the correctness of your optimized code against the provided examples.
4. **Length:** Focus on achieving the shortest possible code while maintaining clarity and correctness. Avoid unnecessary complexity that does not contribute to shortening.
5. **Output Requirements:** Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
Your output wrapped in a single <answer_begin>...</answer_end> block, and must follow the following specified format:
<answer_begin>
Applied Tricks: [List the tricks you applied]
```python
def p(g):
    # Your optimized code with tricks applied
    pass
```
</answer_end>

If NO tricks can safely improve this code, respond with exactly:
<answer_begin>
NO_APPLICABLE_TRICKS
</answer_end>

Now follow the above instructions, find and apply all the useful tricks and provide your optimized code.
"""
