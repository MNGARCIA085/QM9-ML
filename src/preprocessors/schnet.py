from .registry import PreprocessorRegistry
from src.preprocessors.base import BasePreprocessor

@PreprocessorRegistry.register("schnet")
class SchNetPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        # SchNet often handles connectivity internally (e.g., using radius graph
        # and pos), so we might not need an explicit transform here, depending on 
        # how the model is implemented. Sticking with None for consistency.
        super().__init__(transform=None, **kwargs)
        
    
    def _format_dataset(self, dataset, is_inference):
        target_col = self.target

        processed = []
        for d in dataset:
            d_new = d.clone()  # <-- keeps all fields: pos, z, edge_index, etc

            if is_inference:
                # remove y safely if exists
                if hasattr(d_new, "y"):
                    del d_new.y
            else:
                d_new.y = d.y[:, target_col].unsqueeze(1)

            processed.append(d_new)

        return processed