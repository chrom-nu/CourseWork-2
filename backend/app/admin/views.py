from sqladmin import ModelView
from backend.app.models.prediction import Prediction
from backend.app.models.feedback import Feedback


class PredictionAdmin(ModelView, model=Prediction):
    name = "Prediction"
    name_plural = "Predictions"
    icon = "fa-solid fa-seedling"

    column_list = [
        Prediction.id,
        Prediction.created_at,
        Prediction.label,
        Prediction.prob_healthy,
        Prediction.prob_sick,
        Prediction.model_version,
        Prediction.image_path,
    ]
    column_searchable_list = [Prediction.id, Prediction.label, Prediction.model_version]
    column_sortable_list = [Prediction.created_at, Prediction.label, Prediction.prob_sick]
    column_default_sort = [(Prediction.created_at, True)]


class FeedbackAdmin(ModelView, model=Feedback):
    name = "Feedback"
    name_plural = "Feedback"
    icon = "fa-solid fa-comments"

    column_list = [
        Feedback.id,
        Feedback.created_at,
        Feedback.prediction_id,
        Feedback.correct_label,
    ]
    column_searchable_list = [Feedback.id, Feedback.prediction_id, Feedback.correct_label]
    column_sortable_list = [Feedback.created_at, Feedback.correct_label]
    column_default_sort = [(Feedback.created_at, True)]
