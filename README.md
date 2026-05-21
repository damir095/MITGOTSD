# B3 — распознавание дорожных знаков

YOLOv8n детектор + EfficientNet-B0 классификатор на 48 классов
(43 немецких GTSRB + 5 русских RTSD). Подписи на русском.

**Полная инструкция по установке и запуску — `INSTRUCTIONS.md`.**

Быстрый старт:
```powershell
# 1. Установить PyTorch с CUDA отдельно (см. INSTRUCTIONS.md §1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Прогон на видео
python -m src.yolo_pipeline --cam путь\к\видео.mp4 --save out.mp4 --no-show --yolo-imgsz 1280

# 3. Прогон с веб-камеры
python -m src.yolo_pipeline --cam 0
```

Дополнительно: `CLAUDE.md` — архитектура и накопленные уроки проекта.
