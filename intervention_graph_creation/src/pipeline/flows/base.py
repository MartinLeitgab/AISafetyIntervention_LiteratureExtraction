from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..local_graph import LocalGraph


class Flow(ABC):
    """Abstract base class for pipeline flows implementing Chain of Responsibility pattern."""
    
    def __init__(self, next_flow: Optional['Flow'] = None):
        self._next_flow = next_flow
    
    def set_next(self, next_flow: 'Flow') -> 'Flow':
        """Set the next flow in the chain and return it for chaining."""
        self._next_flow = next_flow
        return next_flow
    
    @abstractmethod
    def process(self, local_graph: 'LocalGraph') -> 'LocalGraph':
        """
        Process the local graph and pass it to the next flow in the chain.
        
        Args:
            local_graph: The LocalGraph instance to process
            
        Returns:
            The processed LocalGraph (either modified or passed through)
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def _call_next(self, local_graph: 'LocalGraph') -> 'LocalGraph':
        """Call the next flow in the chain if it exists."""
        if self._next_flow is not None:
            return self._next_flow.process(local_graph)
        return local_graph
