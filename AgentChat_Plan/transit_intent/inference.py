
# file: transit_intent/inference.py
"""
Inference module: loads pipelines and provides predict() function.
"""
from transformers import pipeline

_intent_clf = None
_slot_tagger = None


def load_models(intent_dir: str = "bert_intent_model",
                slot_dir: str = "bert_slot_model",
                aggregation_strategy: str = "simple"):
    """
    Load or reload the intent and slot pipelines.

    Args:
        intent_dir: Path to the saved intent model/tokenizer.
        slot_dir: Path to the saved slot model/tokenizer.
        aggregation_strategy: How to merge token-level predictions into spans.
    """
    global _intent_clf, _slot_tagger
    _intent_clf = pipeline(
        "text-classification",
        model=intent_dir,
        tokenizer=intent_dir,
    )
    _slot_tagger = pipeline(
        "token-classification",
        model=slot_dir,
        tokenizer=slot_dir,
        aggregation_strategy=aggregation_strategy,
    )


def predict(text: str) -> dict:
    """
    Perform intent classification and entity extraction on input text.

    Args:
        text: input utterance.

    Returns:
        dict with keys:
          - intent: dict { 'label': str, 'score': float }
          - entities: dict mapping slot names to extracted text
    """
    if _intent_clf is None or _slot_tagger is None:
        # lazy load with defaults
        load_models()

    intent_pred = _intent_clf(text)[0]
    slot_preds = _slot_tagger(text)
    # convert list of spans to dict
    entities = {
        span['entity_group'].split('-')[-1]: span['word']
        for span in slot_preds
    }
    return {"intent": intent_pred, "entities": entities}


