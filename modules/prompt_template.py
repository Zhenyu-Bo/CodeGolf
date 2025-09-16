"""
Prompt templates for Golf Agent
"""

# Variant generation prompt template
VARIANT_GENERATION_PROMPT = """You are an expert Python programmer specializing in code golf for Google Code Golf 2025 and abstract reasoning puzzles.

**Task Description:**
You will be given a `task_id` and a series of input/output pairs.
-   **Input:** A grid, represented as a 2D list of integers (`list[list[int]]`).
-   **Output:** A transformed grid, also a 2D list of integers.

Your task is to analyze the given problem and generate {n_variants} COMPLETELY DIFFERENT algorithmic approaches to solve it.

## Problem Analysis

### Step 1: Analyze Input-Output Examples
Here are the input-output examples of the task:
{examples_str}

Your code should implement the function `p(g)` where `g` is the input grid (list of lists of integers), and return the modified output grid.

### Step 2: Analyze Generator Code
Here is the code that generates the input-output examples. Analyze it carefully to understand the underlying transformation rules:
```python
{generator_code}
```

**Analysis Instructions:**
- Identify what patterns the generator creates in the input
- Understand how the transformation from input to output works
- Look for implicit constraints or conditions in the generation process
- Identify key parameters and their relationships

### Step 3: Analyze Current Solution
Here is the current solution code which has been proved correct:
```python
{initial_solution}
```

**Analysis Instructions:**
- Understand the core logic and transformation rule implemented
- Identify the key algorithmic steps and data structures used
- Look for potential redundancies or optimization opportunities
- Understand which parts of the generator constraints this solution leverages

### Step 4: Identify Core Transformation Rules
Based on your analysis above:
1. **Primary Rule:** [State the main transformation rule in one clear sentence]
2. **Edge Cases:** [Identify any special cases or boundary conditions]
3. **Constraints:** [List any implicit constraints from the generator]
4. **Key Insights:** [Important observations that affect implementation]

## Variant Generation Requirements

Generate {n_variants} fundamentally different approaches that:
1. **Implement the SAME transformation rule** but use different algorithms
2. **Focus on algorithmic diversity**, not just code golf optimizations
3. **Each approach must be complete and functional**
4. **Preserve exact functionality** while using different strategies

## Strategy Diversity Guidelines

### GOOD Diversity Examples:
- **Data Structure Change:** Original uses dictionary → Alternative uses 2D array indexing
- **Processing Order:** Original processes row-by-row → Alternative processes column-by-column
- **Algorithm Paradigm:** Original uses iterative approach → Alternative uses recursive approach
- **Coordinate System:** Original uses (row,col) → Alternative uses linear indexing
- **Detection Method:** Original scans patterns → Alternative uses mathematical formulas
- **Construction Method:** Original builds incrementally → Alternative creates full result then modifies

### BAD Diversity Examples (Avoid):
- Same algorithm with different variable names only
- Same logic with minor syntax variations
- Same approach with different loop styles only
- Simple code golf tricks without algorithmic changes

## Implementation Strategies to Consider

1. **Different Iteration Patterns:**
   - Row-major vs column-major traversal
   - Spiral or diagonal traversal
   - Backwards vs forwards iteration

2. **Different Data Structures:**
   - Lists vs dictionaries vs sets
   - Single-pass vs multi-pass processing
   - In-place modification vs creating new grid

3. **Different Algorithmic Approaches:**
   - Scanning and replacement
   - Mathematical calculation
   - Pattern matching and transformation
   - State machine or rule-based processing

4. **Different Coordinate Systems:**
   - (row, col) indexing
   - Linear indexing with math conversion
   - Relative positioning
   - Offset-based calculations

## Output Format

For each variant, use this EXACT format:

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

## Critical Requirements:
- Each variant must implement the SAME transformation rule correctly
- Code must be syntactically correct and complete
- Focus on algorithmic diversity, not micro-optimizations
- Each approach should be genuinely different in its core strategy
- Test your logic mentally against the examples before providing code

{shortest_hint}

Begin your analysis and variant generation now:"""

# Optimization prompt template
OPTIMIZATION_PROMPT = """You are an expert Python programmer specializing in code golf for Google Code Golf 2025.

**Task Description:**
Your task is to optimize the given Python code to reduce byte count while maintaining EXACT functionality.

## Problem Context

### Task Examples:
Here are the input-output examples of the task:
{examples_str}

### Generator Code:
Here is the code that generates the input-output examples. Analyze it carefully to understand the underlying transformation rules:
```python
{generator_code}
```

**Note that the generator code is very important to the task because it may contain implicit constraints and conditions that you can use to reduce some unnecessary processing in the provided code so that shorten it.**



**Analysis Instructions:**
- Identify what patterns the generator creates in the input
- Understand how the transformation from input to output works
- Look for implicit constraints or conditions in the generation process
- Identify key parameters and their relationships

### Current Code to Optimize:
Here is the current solution code that needs optimization:
```python
{code}
```

{history_str}

## Optimization Strategy

### Step 1: Analyze Problem
1. **Understand the core logic** and transformation rule implemented
2. **Carefully read the generator code and analyze if it contains any implicit constraints or conditions that you can use to reduce some unnecessary processing in the provided code.**
3. **Identify redundancies** or unnecessary operations
4. **Trace variable usage** and data flow
5. **Check alignment with generator constraints** to avoid redundant processing

### Step 2: Systematic Optimization Process

#### A. Algorithmic Optimizations
- **Eliminate redundant computations** or data structures
- **Combine multiple passes** into single iterations where possible
- **Simplify logic flow** by removing unnecessary conditions
- **Use mathematical shortcuts** instead of iterative approaches

#### B. Python-Specific Optimizations
- **Variable name reduction:** Use single letters (i,j,r,c,x,y,z,n,m,k,v,w,h)
- **Walrus operator:** `if x:=expr():` instead of `x=expr(); if x:`
- **Tuple unpacking:** `a,b=b,a` for swaps, `*args` for expansion
- **Comprehensions:** Replace loops with list/dict/set comprehensions when shorter
- **Boolean shortcuts:** `x and y or z` instead of `if x: y else: z` (when safe)
- **Chained comparisons:** `a<b<c` instead of `a<b and b<c`
- **Built-in functions:** `sum()`, `max()`, `min()`, `any()`, `all()`, `enumerate()`

#### C. Syntactic Optimizations
- **Remove unnecessary spaces:** Around operators, after commas (where safe)
- **Remove unnecessary parentheses:** In expressions and conditions
- **Combine statements:** Use semicolons where appropriate
- **Shorter alternatives:** `x//1` for int(), `x or y` for default values

#### D. Advanced Golf Techniques
- **String/sequence operations:** Slicing tricks, join operations
- **Mathematical shortcuts:** Bitwise operations, modulo arithmetic
- **Lambda functions:** For simple transformations
- **Generator expressions:** When memory efficient and shorter

### Step 3: Verify Optimization
1. **Functionality check:** Ensure the optimized code produces identical outputs
2. **Edge case verification:** Test against boundary conditions and edge cases
3. **Byte count measurement:** Use `len(code.encode())` to verify reduction
4. **Alignment with generator constraints:** Ensure no redundant processing

## Output Requirements

Provide ONLY the optimized code in a clean Python code block:

```python
def p(g):
    # Your optimized implementation
    # Focus on shortest byte count while preserving functionality
    pass
```

**Optimization Objective:** Achieve maximum byte count reduction while maintaining exact functionality and correctness."""

# Tricks application prompt template
TRICKS_PROMPT = """Apply specific code golf tricks to make this code shorter while maintaining correctness.

## Current Code:
```python
{code}
```

## Available Golf Tricks:
{tricks_str}

## Instructions:
1. Analyze which tricks can be safely applied to the current code and shorten the current code indeed
2. Apply applicable tricks step by step
3. Ensure the function still works correctly
4. Provide the final optimized code and explain which tricks were applied

Apply the tricks and provide your result:"""

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

## Systematic Debugging Process

### Step 1: Error Classification and Analysis

#### A. Identify Error Type:
- **Syntax Error:** Missing colons, parentheses, indentation issues, invalid syntax
- **Runtime Error:** IndexError, KeyError, TypeError, AttributeError, ValueError
- **Logic Error:** Wrong algorithm implementation, incorrect edge case handling
- **Test Failure:** Code runs but produces incorrect output for some inputs

#### B. Analyze Error Context:
- **Where does the error occur?** (line, function, operation)
- **What conditions trigger it?** (specific inputs, edge cases)
- **What was the intended behavior?** (based on the strategy)

### Step 2: Systematic Error Resolution

#### A. For Syntax Errors:
- **Check indentation:** Consistent spacing (4 spaces per level)
- **Check colons:** After if/for/while/def statements
- **Check parentheses/brackets:** Proper matching and nesting
- **Check variable names:** Valid Python identifiers

#### B. For Runtime Errors:
- **IndexError:** Add bounds checking, handle empty sequences
  - Use `if i < len(list):` before accessing `list[i]`
  - Check grid dimensions before accessing `g[r][c]`
- **KeyError:** Use `.get()` method or check key existence
  - Replace `dict[key]` with `dict.get(key, default)`
- **TypeError:** Ensure correct data types and handle None values
  - Check for None before operations
  - Ensure consistent data types in operations

#### C. For Logic Errors:
- **Re-examine the core strategy:** {strategy}
- **Trace through algorithm** with simple test cases
- **Check edge cases:** Empty grids, single cells, boundary conditions
- **Verify transformation logic** against expected behavior

### Step 3: Strategy Preservation Guidelines

**Critical Requirements:**
- **Maintain the same core approach:** {strategy}
- **Preserve algorithmic structure** and data flow
- **Keep the same data structures** and processing order
- **Fix only the specific error** without changing the fundamental logic

**What TO Fix:**
- Syntax errors and typos
- Index bounds and safety checks
- Type errors and None handling
- Off-by-one errors in loops or indexing

**What NOT TO Change:**
- Core algorithmic approach
- Overall code structure
- Variable usage patterns (unless causing errors)
- The fundamental strategy implementation

### Step 4: Verification Process

After fixing, mentally verify:
1. **Syntax correctness:** Code parses without syntax errors
2. **Runtime safety:** No index/key/type errors on valid inputs
3. **Logic correctness:** Implements the intended transformation
4. **Strategy preservation:** Still follows the original approach
5. **Edge case handling:** Works with boundary conditions

## Common Fixes by Error Pattern

### IndexError Patterns:
```python
# BEFORE (Error-prone):
g[r][c] = value
# AFTER (Safe):
if 0 <= r < len(g) and 0 <= c < len(g[0]):
    g[r][c] = value
```

### KeyError Patterns:
```python
# BEFORE (Error-prone):
value = dict[key]
# AFTER (Safe):
value = dict.get(key, default)
```

### TypeError Patterns:
```python
# BEFORE (Error-prone):
if item:  # item might be None
    process(item)
# AFTER (Safe):
if item is not None:
    process(item)
```

## Output Requirements

Provide ONLY the corrected code that:
- **Fixes the specific error** mentioned above
- **Preserves the core strategy:** {strategy}
- **Maintains the same algorithmic approach**
- **Ensures function signature** remains `def p(g):`

```python
def p(g):
    # Your corrected implementation
    # Preserves strategy: {strategy}
    # Fixes error: {error_summary}
    pass
```

Focus on minimal, targeted fixes that resolve the error while preserving the original algorithmic approach."""

# Knowledge base tricks scanning prompt template
KNOWLEDGE_BASE_TRICKS_PROMPT = """You are an expert Python code golf specialist with deep knowledge of optimization techniques.

Your task is to analyze the given code and apply proven code golf optimizations from your knowledge base.

## Code Analysis Target

### Current Code:
```python
{code}
```

## Systematic Optimization Analysis

### Step 1: Code Structure Analysis
1. **Identify patterns** that can be optimized using known techniques
2. **Map code constructs** to potential golf transformations
3. **Analyze variable usage** and scope optimization opportunities
4. **Detect algorithmic redundancies** or simplification opportunities

### Step 2: Knowledge Base Application

#### A. Variable and Namespace Optimization
**Target:** Reduce character count through naming and scope
- **Long variable names → Single letters:** Use `i,j,k,r,c,x,y,z,n,m,v,w,h`
- **Reduce variable count:** Combine or eliminate temporary variables
- **Reuse variables:** Use same variable for different purposes when scope allows
- **Global variables:** Use when it reduces total character count

#### B. Expression and Operator Optimization
**Target:** Compact expressions and leverage Python operator features
- **Walrus operator:** `x=expr(); if x:` → `if x:=expr():`
- **Chained comparisons:** `a<b and b<c` → `a<b<c`
- **Boolean algebra:** `not(not x)` → `bool(x)`, `x==True` → `x`
- **Tuple indexing:** `if cond: a else b` → `(b,a)[cond]` (when safe)
- **Mathematical shortcuts:** `x//1` for int(), `x**2` vs `x*x`
- **Bitwise operations:** When applicable and shorter

#### C. Loop and Iteration Optimization
**Target:** Reduce iteration overhead and combine operations
- **List comprehensions:** Replace simple loops when shorter
- **Generator expressions:** For memory efficiency and syntax reduction
- **Enumerate vs range:** Use `enumerate()` instead of manual indexing
- **Zip operations:** Parallel iteration over multiple sequences
- **All/any shortcuts:** Replace boolean accumulation loops
- **Flatten nested loops:** Convert to single loop with math when possible

#### D. Built-in Function Leverage
**Target:** Use Python built-ins instead of manual implementation
- **Aggregation functions:** `sum()`, `max()`, `min()` instead of loops
- **Sequence operations:** `map()`, `filter()` for transformations
- **String operations:** `str.join()`, slicing tricks
- **Set operations:** For unique elements, intersections
- **Sorting shortcuts:** `sorted()` with key functions

#### E. Data Structure Optimization
**Target:** Choose optimal data structures for minimal code
- **List vs dict vs set:** Choose based on access patterns
- **In-place modification:** Modify input instead of creating new structures
- **Tuple unpacking:** `a,b,c = sequence` for multiple assignment
- **Default dict patterns:** Use `dict.get()` or `setdefault()`

#### F. Control Flow Optimization
**Target:** Simplify conditional logic and flow control
- **Early returns:** Reduce nesting with guard clauses
- **Ternary operators:** `x if condition else y` when appropriate
- **Short-circuit evaluation:** Leverage `and`/`or` for control flow
- **Exception handling:** Use try/except when shorter than checks

### Step 3: Space and Syntax Optimization
**Target:** Remove unnecessary characters while maintaining readability
- **Remove unnecessary spaces:** Around operators, after commas (where safe)
- **Remove unnecessary parentheses:** In expressions and conditions
- **Combine statements:** Use semicolons where appropriate
- **Line breaks:** Optimize for minimal total characters

### Step 4: Advanced Golf Patterns
**Target:** Apply sophisticated optimization patterns
- **String manipulation tricks:** Slicing, indexing, formatting shortcuts
- **Lambda optimizations:** For simple function definitions
- **Recursive patterns:** When they reduce code vs iterative approaches
- **Mathematical formulas:** Replace algorithmic computation when possible

## Application Strategy

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

## Output Requirements

Apply ALL applicable optimizations from your knowledge base and provide ONLY the final optimized code:

```python
def p(g):
    # Your fully optimized code
    # Apply maximum safe optimizations
    pass
```

**Optimization Goal:** Achieve maximum byte count reduction while preserving exact functionality and correctness."""

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

## Output Instructions:
If you can apply any tricks to reduce the code size:
```python
def p(g):
    # Your optimized code with tricks applied
    pass
```

If NO tricks can safely improve this code, respond with exactly:
NO_APPLICABLE_TRICKS

Apply the most effective applicable tricks:"""
