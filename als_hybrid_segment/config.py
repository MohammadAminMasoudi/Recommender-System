from dataclasses import dataclass

USER_COL = "user_id"
ITEM_COL = "MatNum"
DATE_COL = "BillingDate"
SEGMENT_COL = "MatName"
RATING_COL = "combined_rating"

@dataclass
class HybridConfig:
    topk: int = 10
    candidates: int = 200
    weight_als: float = 0.6
    weight_cooc: float = 0.2
    weight_pop: float = 0.2
    lambda_seg: float = 0.6
    decay_pop: float = 0.9
    temp_softmax: float = 1.0
    seg_strength: float = 1.0
