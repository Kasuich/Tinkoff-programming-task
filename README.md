# Tinkoff-programming-task

#### 1. Неоходимые библиотеки находятся в requirements.txt

#### 2. Обучение модели:
  
  ```
  python train.py files plagiat1 plagiat2 --model model.pkl
  ```

#### 3. Определение плагиата:

  ```
  python compare.py input.txt scores.txt --model model.pkl
  ```
