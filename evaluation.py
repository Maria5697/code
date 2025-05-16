import re
import pandas as pd
import sys
from rurage import RAGEvaluator, RAGESetConfig, RAGEModelConfig

sys.path.insert(0, "../")


validation_set = pd.read_excel('examples/set_2_upd_with answer.xlsx')
print("Dataset loaded")

# предобработка
def parse_string(raw_str):
    matches = re.findall(r'\(\\"(.*?)\\",\s*([0-9.]+)\)', raw_str, re.DOTALL)
    texts = []
    scores = []

    for raw_text, raw_score in matches:
        cleaned = raw_text.replace('\\"', '"').replace('""', '"').replace('\\n', '\n').strip()
        texts.append(f'"({cleaned})"')
        scores.append(float(raw_score))

    context_text = "{" + ",".join(texts) + "}"
    return pd.Series([context_text, scores])

validation_set[['context_text', 'score']] = validation_set['context'].apply(parse_string)

validation_set['initial_question'] = validation_set['initial_question'].astype(str)
validation_set['golden_answer'] = validation_set['golden_answer'].astype(str)
validation_set['response'] = validation_set['response'].astype(str)

# Конфигурация модели оценки
models_cfg = [RAGEModelConfig(context_col="context_text", answer_col="response")]
validation_set_cfg = RAGESetConfig(
    golden_set=validation_set,
    question_col="initial_question",
    golden_answer_col="golden_answer",
    models_cfg=models_cfg,
)

# Запуск оценки
rager = RAGEvaluator(golden_set_cfg=validation_set_cfg)
print("Running comprehensive evaluation...")
comprehensive_report = rager.comprehensive_evaluation(pointwise_report=True)
print("Evaluation complete")
print(comprehensive_report)

correctness_pointwise = comprehensive_report["correctness_pointwise"][0]
faithfulness_pointwise = comprehensive_report["faithfulness_pointwise"][0]
relevance_pointwise = comprehensive_report["relevance_pointwise"][0]


def add_prefix_except(df, prefix, exclude_cols):
    return df.rename(columns={col: f"{prefix}{col}" if col not in exclude_cols else col for col in df.columns})

exclude_cols = ["initial_question", "golden_answer", "context_text", "response"]
correctness_pointwise = add_prefix_except(correctness_pointwise, "correctness_", exclude_cols)
faithfulness_pointwise = add_prefix_except(faithfulness_pointwise, "faithfulness_", exclude_cols)
relevance_pointwise = add_prefix_except(relevance_pointwise, "relevance_", exclude_cols)

# Объединение метрик по строкам
pointwise_metrics = pd.concat([
    correctness_pointwise,
    faithfulness_pointwise.drop(columns=exclude_cols),
    relevance_pointwise.drop(columns=exclude_cols)
], axis=1)

# Добавление исходных столбцов
pointwise_metrics["initial_question"] = validation_set["initial_question"]
pointwise_metrics["score"] = validation_set["score"]
pointwise_metrics["topic"] = validation_set["topic"]
pointwise_metrics["subtopic"] = validation_set["subtopic"]

# Сохраняем результат
pointwise_metrics.to_excel("validation_set_with_metrics.xlsx", index=False)
print("метрики сохранены в validation_set_with_metrics.xlsx")