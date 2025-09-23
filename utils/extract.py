"""
Utility functions for extracting content from LLM responses
"""

import re
from typing import List, Optional, Tuple


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from response, considering answer_begin/answer_end format"""
    # First try to extract content from <answer_begin>...</answer_end> blocks
    answer_block_pattern = r'<answer_begin>(.*?)<answer_end>'
    answer_match = re.search(answer_block_pattern, response, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        # Fallback to full response if no answer block found
        answer_content = response
    
    # Now extract code from markdown blocks within the answer content
    patterns = [
        r"```(?:python|py|python3)\s+(.*?)\s*```",
        r"```\s*(.*?)\s*```"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, answer_content, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            if 'def p(' not in code and 'def solve(' in code:
                code = code.replace('def solve(', 'def p(')
            elif 'def ' not in code:
                lines = code.split('\n')
                indented_lines = [' ' + line if line.strip() else line for line in lines]
                code = 'def p(g):\n' + '\n'.join(indented_lines)
            return code
    return None


def extract_variants(response: str) -> List[Tuple[str, str]]:
    """Extract multiple code variants from LLM response"""
    variants = []
    
    # First extract content from <answer_begin>...</answer_end> block
    answer_block_pattern = r'<answer_begin>(.*?)<answer_end>'
    answer_match = re.search(answer_block_pattern, response, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        # Fallback to full response if no answer block found
        answer_content = response
    
    # Look for sections like "### Variant 1:", "### Variant 2:", etc.
    sections = re.split(r'(?:##|###)\s*(?:Strategy|Variant|Approach)\s*\d+[:\s]*', answer_content)
    
    for section in sections[1:]:  # Skip first empty section
        # Extract strategy description from **Strategy:** field
        strategy_match = re.search(r'\*\*(?:Strategy|Core Strategy):\*\*\s*(.*?)(?:\*\*|```|$)', section, re.DOTALL)
        if strategy_match:
            strategy = strategy_match.group(1).strip()
            # Clean up the strategy text
            strategy = re.sub(r'\s+', ' ', strategy)
            strategy = strategy.split('\n')[0]  # Take first line only
        else:
            # Fallback: extract first meaningful line
            lines = section.strip().split('\n')
            strategy = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith('**') and not line.startswith('```'):
                    strategy = line
                    break
            if not strategy:
                strategy = "Unknown Strategy"
        
        # Extract code
        code = extract_code(section)
        if code and strategy:
            variants.append((code, strategy))
            
        # Also check for variant end marker
        if "**End of Variant" in section:
            continue
    
    return variants


def extract_answer_content(response: str) -> str:
    """Extract content from <answer_begin>...</answer_end> blocks, fallback to full response"""
    answer_block_pattern = r'<answer_begin>(.*?)<answer_end>'
    answer_match = re.search(answer_block_pattern, response, re.DOTALL)
    
    if answer_match:
        return answer_match.group(1).strip()
    else:
        return response


def extract_strategies_from_section(section: str) -> str:
    """Extract strategy description from a section of text"""
    strategy_match = re.search(r'\*\*(?:Strategy|Core Strategy):\*\*\s*(.*?)(?:\*\*|```|$)', section, re.DOTALL)
    if strategy_match:
        strategy = strategy_match.group(1).strip()
        # Clean up the strategy text
        strategy = re.sub(r'\s+', ' ', strategy)
        strategy = strategy.split('\n')[0]  # Take first line only
        return strategy
    else:
        # Fallback: extract first meaningful line
        lines = section.strip().split('\n')
        strategy = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith('**') and not line.startswith('```'):
                strategy = line
                break
        if not strategy:
            strategy = "Unknown Strategy"
        return strategy
