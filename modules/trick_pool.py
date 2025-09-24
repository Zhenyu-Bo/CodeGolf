"""
Trick pool module: Manage storage, sampling, and updates of code optimization tricks
"""

import json
import os
import time
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Trick:
    """Represents a code optimization trick (simplified version)"""
    description: str  # Description of the trick
    sub_problem: str  # Sub-problem extracted from the task
    input_code: str   # Code before applying the trick
    output_code: str  # Code after applying the trick
    success_count: int = 0  # Number of successful applications
    tasks_applied: Set[int] = field(default_factory=set)  # Tasks where this trick was applied
    
    def __post_init__(self):
        if isinstance(self.tasks_applied, list):
            self.tasks_applied = set(self.tasks_applied)
    
    @property
    def length_reduction(self) -> int:
        """Length reduction amount"""
        return len(self.input_code) - len(self.output_code)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'description': self.description,
            'sub_problem': self.sub_problem,
            'input_code': self.input_code,
            'output_code': self.output_code,
            'success_count': self.success_count,
            'tasks_applied': list(self.tasks_applied)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trick':
        """Create a Trick object from dictionary"""
        return cls(**data)

class TrickPoolManager:
    """Class for managing trick pool (improved sampling logic)"""
    
    def __init__(self, save_path: str = "trick_pool.json", max_sample_threshold: int = 50):
        self.tricks: List[Trick] = []
        self.save_path = save_path
        self.max_sample_threshold = max_sample_threshold  # Sampling threshold
        self._current_sample_index = 0  # Current sampling position
        self._sampled_count = 0  # Total number of samples taken
        self.load_tricks()
        self._sort_tricks()
    
    def _sort_tricks(self) -> None:
        """Sort tricks by success count in descending order"""
        self.tricks.sort(key=lambda t: t.success_count, reverse=True)
        self._current_sample_index = 0  # Reset sampling position
        self._sampled_count = 0  # Reset sampling count
    
    def add_trick(self, trick: Trick) -> None:
        """Add new trick to the pool"""
        # Check if a similar trick already exists
        for existing_trick in self.tricks:
            if (existing_trick.input_code.strip() == trick.input_code.strip() and 
                existing_trick.output_code.strip() == trick.output_code.strip()):
                # Merge trick information
                existing_trick.success_count += trick.success_count
                existing_trick.tasks_applied.update(trick.tasks_applied)
                self.save_tricks()
                self._sort_tricks()  # Re-sort
                return
        
        self.tricks.append(trick)
        self.save_tricks()
        self._sort_tricks()  # Re-sort
    
    def sample_tricks(self, n: int, exclude_tasks: Set[int] = None) -> List[Trick]:
        """
        Improved sampling logic:
        1. Sort by success count in descending order
        2. Sequential sampling to avoid duplicates
        3. Skip tricks already used in current task
        4. Stop after reaching threshold
        """
        if exclude_tasks is None:
            exclude_tasks = set()
        
        if not self.tricks or self._sampled_count >= self.max_sample_threshold:
            return []
        
        sampled_tricks = []
        start_index = self._current_sample_index
        
        # Start sampling from current position
        for i in range(len(self.tricks)):
            if len(sampled_tricks) >= n:
                break
            if self._sampled_count >= self.max_sample_threshold:
                break
                
            trick_index = (start_index + i) % len(self.tricks)
            trick = self.tricks[trick_index]
            
            # Skip tricks already used in excluded tasks
            if exclude_tasks.intersection(trick.tasks_applied):
                continue
            
            sampled_tricks.append(trick)
            self._sampled_count += 1
        
        # Update sampling position
        self._current_sample_index = (start_index + len(sampled_tricks)) % len(self.tricks)
        
        # Reset counter if completed one round
        if self._current_sample_index == 0 and len(sampled_tricks) < n:
            self._sampled_count = 0
        
        return sampled_tricks
    
    def reset_sampling(self) -> None:
        """Reset sampling state, usually called at the start of new optimization rounds"""
        self._current_sample_index = 0
        self._sampled_count = 0
        self._sort_tricks()  # Re-sort to reflect latest success counts
    
    def update_trick_success(self, trick: Trick, task_id: int) -> None:
        """Update trick success usage"""
        trick.success_count += 1
        trick.tasks_applied.add(task_id)
        self.save_tricks()
        # Note: Don't immediately re-sort here as it might affect ongoing sampling
        # Can call reset_sampling() at appropriate times (like end of rounds) to re-sort
    
    def get_stats(self) -> Dict:
        """Get trick pool statistics"""
        if not self.tricks:
            return {
                "total": 0, 
                "avg_success_count": 0, 
                "most_successful": None,
                "sampling_progress": f"0/{self.max_sample_threshold}"
            }
        
        total = len(self.tricks)
        avg_success = sum(t.success_count for t in self.tricks) / total
        most_successful = max(self.tricks, key=lambda t: t.success_count)
        
        return {
            "total": total,
            "avg_success_count": avg_success,
            "most_successful": {
                "description": most_successful.description,
                "success_count": most_successful.success_count,
                "length_reduction": most_successful.length_reduction
            },
            "sampling_progress": f"{self._sampled_count}/{self.max_sample_threshold}",
            "current_position": f"{self._current_sample_index}/{total}"
        }
    
    def save_tricks(self) -> None:
        """Save trick pool to file"""
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                data = {
                    'tricks': [trick.to_dict() for trick in self.tricks],
                    'saved_at': datetime.now().isoformat(),
                    'max_sample_threshold': self.max_sample_threshold
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save tricks: {e}")
    
    def load_tricks(self) -> None:
        """Load trick pool from file"""
        if not os.path.exists(self.save_path):
            return
        
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.tricks = [Trick.from_dict(trick_data) for trick_data in data.get('tricks', [])]
                # Load saved threshold settings
                if 'max_sample_threshold' in data:
                    self.max_sample_threshold = data['max_sample_threshold']
        except Exception as e:
            print(f"Failed to load tricks: {e}")
            self.tricks = []