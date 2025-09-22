# Avito-Test

КАК ЗАПУСТИТЬ

1) (опционально) создать окружение
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
2) установить зависимости
   pip install < requirements.txt
3) запустить инференс

   python inference.py -i dataset.csv -o submission.csv

   python inference.py -i dataset_XXXX.txt -о submission.csv

ВЫХОД
submission.csv — копия входного файла с добавленной колонкой predicted_positions
в формате строки: [5, 8, 13]

ПОДХОД (кратко)

- Корпус: русская Википедия. Из предложений строим пары (text_no_spaces, boundaries).
- Фичи: символьное окно ±5, классы символов/переходы, регистры, локальные токены.
  Используется лексикон частот из prepared_corpus/unigram_freq.tsv (униграммы).
- Модель: логистическая регрессия (SGDClassifier) с hashing-trick; баланс классов —
  даунсэмпл негативов + веса.
- Порог: выбран по F1 на валидации; берётся из artifacts_boundary/best_threshold.txt.
- Артефакты инференса: artifacts_boundary/boundary_sgd.joblib, artifacts_boundary/best_threshold.txt,
  artifacts_boundary/config_boundary.json, prepared_corpus/unigram_freq.tsv.
