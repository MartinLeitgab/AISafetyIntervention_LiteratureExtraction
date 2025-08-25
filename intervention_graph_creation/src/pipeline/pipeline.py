from typing import List, Optional

from .flows.base import Flow
from .local_graph import LocalGraph


class Pipeline:
    """Main pipeline class that manages a chain of flows using Chain of Responsibility pattern."""
    
    def __init__(self):
        self._start_flow: Optional[Flow] = None
        self._curr_flow: Optional[Flow] = None
        self._chain: List[Flow] = []  # For viewing purposes only
    
    def __repr__(self):
        return f"Pipeline(start_flow={self._start_flow}, curr_flow={self._curr_flow}, chain={self._chain})"
    
    def __str__(self):
        return self.__repr__()

    def push_node(self, flow: Flow) -> 'Pipeline':
        """
        Add a flow node to the pipeline chain.
        
        Args:
            flow: The Flow instance to add
            
        Returns:
            Self for method chaining
        """
        # Validation check
        if not isinstance(flow, Flow):
            raise TypeError(f"Expected Flow instance, got {type(flow)}")
        
        # Handle start_flow and curr_flow
        if self._start_flow is None:
            self._start_flow = flow
            self._curr_flow = flow
        else:
            self._curr_flow.set_next(flow)
            self._curr_flow = flow
        
        # Add to chain for viewing purposes
        self._chain.append(flow)
        
        return self
    
    def process(self, output_path: str) -> LocalGraph:
        """
        Process the pipeline starting from the output JSONL file.
        
        Args:
            output_path: Path to the output JSONL file
            
        Returns:
            The processed LocalGraph
        """
        if self._start_flow is None:
            raise ValueError("No flows have been added to the pipeline")
        
        # The first flow will handle the input (could be EmbedderFlow for JSONL or any other flow)
        return self._start_flow.process(output_path)

    def clear(self) -> None:
        """Clear all flows from the pipeline."""
        self._start_flow = None
        self._curr_flow = None
        self._chain.clear()
