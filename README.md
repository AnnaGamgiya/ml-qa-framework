
    # ML QA Framework

    Учебный фреймворк для тестирования ML-моделей методами классического QA.  
    Показывает, как применять эквивалентное разбиение, граничные значения, попарное тестирование (pairwise) и таблицы решений к модели машинного обучения.

    Фреймворк написан на Python, использует Streamlit для дашборда и открытый датасет VK-LSVD.

    ## Что внутри

    - **5 тестов**: Segmented AUC, Boundary Test, Decision Table, Pairwise Test, Stability Test.
    - **Дашборд** на Streamlit с пояснениями и графиками.
    - **Конфигурационный файл** (`config.yaml`) для адаптации под свои данные.
    - **Справочник терминов** прямо в дашборде.

    ## Быстрый старт

    1. Клонируйте репозиторий:
       ```
       git clone https://github.com/AriadnaQA/ml-qa-framework.git
       cd ml-qa-framework
       ```

    2. Создайте и активируйте виртуальное окружение:
       ```
       python -m venv venv
       venv\Scripts\activate
       ```

    3. Установите зависимости:
       ```
       pip install -r requirements.txt
       ```

    4. Скачайте демо-датасет:
       ```
       python download_data.py
       ```

    5. Обучите пайплайн:
       ```
       python train_pipeline.py
       ```

    6. Запустите дашборд:
       ```
       streamlit run app.py
       ```
       Откройте [http://localhost:8501](http://localhost:8501)

    ## Адаптация под свои данные

    Все настройки в `config.yaml`. Можно менять:
    - Пути к данным
    - Пороги (AUC, PSI, размер группы, эффект, p-value)
    - Бизнес-гипотезы
    - Список признаков
    - Параметры для pairwise-теста

    После изменений переобучите пайплайн и перезапустите дашборд.

    ## Файлы

    - `config.yaml` — Настройки
    - `download_data.py` — Скачивание датасета
    - `train_pipeline.py` — Обучение модели
    - `preprocessing.py` — Кодирование авторов
    - `ml_tests.py` — Тесты
    - `app.py` — Дашборд

    ## Требования

    Python 3.10+, Streamlit, Polars, Pandas, Scikit-learn, SciPy, AllPairsPy, Plotly, HuggingFace Hub, PyYAML
    ```
6.  Нажмите зелёную кнопку **Commit changes...**.

