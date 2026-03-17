from .Evaluator import Evaluator

def create_evaluator(model_provider, info_cats, metric_1='acc', metric_2='rouge1'):
    return Evaluator(model_provider, info_cats, metric_1, metric_2)
