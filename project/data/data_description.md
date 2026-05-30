# Данные проекта

Для обучения моделей был использован датасет TrashNet, оригинал можно найти тут:  
- <https://github.com/garythung/trashnet> 

Он также доступен на Hugging Face:  
- <https://huggingface.co/datasets/garythung/trashnet>  

Я воспользовался зеркалом с Kaggle для удобства:  
- <https://www.kaggle.com/datasets/feyzazkefe/trashnet>

>### Краткий обзор
>(Информация была получена из EDA анализа в `notebooks/exp01_eda.ipynb`)  
>Размер датасета: 2527  
>Классы в датасете:  cardboard, glass, metal, paper, plastic, trash  
>Процентное соотношение: 15.95%, 19.83%, 16.22%, 23.51%, 19.07%, 5.42%  

`/trashnet-sample` содержит по 10 изображений на каждый класс.  
Внутри также лежит `trashnet_sample.zip`, который можно использовать для FastAPI.

>### Примечание
>`/trashnet-sample` и `test.jpg` используются в **tests**.  
>Если их удалить, то `pytest tests` не пройдут проверку.



Лицензия датасета:
```
MIT License

Copyright (c) 2017 Gary Thung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```