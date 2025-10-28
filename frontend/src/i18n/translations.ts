export type Locale = 'en' | 'ru';

export const translations = {
  en: {
    // Navigation
    playground: 'Playground',
    training: 'Training',

    // Playground
    playground_title: 'Math HWR — Playground',
    playground_subtitle: 'Draw mathematical expressions or load an image to recognize',
    brush: 'Brush',
    eraser: 'Eraser',
    size: 'Size',
    zoom: 'Zoom',
    reset_view: 'Reset View',
    open_file: 'Open File',
    clear: 'Clear',
    recognize: 'Recognize',
    recognizing: 'Recognizing...',

    // Results
    recognized_text: 'Recognized Text',
    copy: 'Copy',
    copied: 'Copied!',
    tokens: 'Tokens',
    preprocessed_image: 'Preprocessed Image',
    inference_time: 'Inference time',

    // Training
    experiments: 'Experiments',
    new_training: 'New Training',
    monitor: 'Monitor',
    models: 'Models',

    // New Training
    new_training_title: 'New Training Run',
    basic_settings: 'Basic Settings',
    advanced_settings: 'Advanced Settings',
    model_type: 'Model Type',
    run_name: 'Run Name',
    batch_size: 'Batch Size',
    epochs: 'Epochs',
    learning_rate: 'Learning Rate',
    seed: 'Seed',
    invert_colors: 'Invert Colors',
    cancel: 'Cancel',
    start_training: 'Start Training',
    creating: 'Creating...',

    // Monitor
    status: 'Status',
    epoch: 'Epoch',
    loss_curves: 'Loss Curves',
    train_loss: 'Train Loss',
    val_loss: 'Val Loss',
    latest_metrics: 'Latest Metrics',
    logs: 'Logs',
    sample_predictions: 'Sample Predictions',
    target: 'Target',
    pred: 'Pred',
    match: 'Match',

    // Experiments
    search_by_name: 'Search by name...',
    all_models: 'All Models',
    all_statuses: 'All Statuses',
    compare: 'Compare',
    no_experiments: 'No experiments yet',
    create_first_run: 'Create Your First Training Run',
    no_matches: 'No experiments match your filters',

    // Models
    models_checkpoints: 'Models & Checkpoints',
    checkpoints: 'Checkpoints',
    select_checkpoint: 'Select a checkpoint to view details',
    info: 'Info',
    quick_test: 'Quick Test',
    checkpoint_details: 'Checkpoint Details',
    type: 'Type',
    path: 'Path',
    metrics_at_save: 'Metrics at Save',
    promote_to_production: 'Promote to Production',
    download: 'Download',
    delete: 'Delete',

    // Quick Test
    canvas_mode: 'Canvas',
    upload_mode: 'Upload',
    select_image: 'Select Image',

    // Comparison
    compare_experiments: 'Compare Experiments',
    loss_curves_comparison: 'Loss Curves Comparison',

    // Status values
    QUEUED: 'Queued',
    RUNNING: 'Running',
    FINISHED: 'Finished',
    FAILED: 'Failed',
    STOPPED: 'Stopped',

    // Messages
    loading: 'Loading...',
    no_logs: 'No logs yet',
    no_metrics: 'No metrics yet',
    no_checkpoints: 'No checkpoints yet',

    // Actions
    view: 'View',
    close: 'Close',
    save: 'Save',
    duplicate: 'Duplicate',
  },

  ru: {
    // Navigation
    playground: 'Площадка',
    training: 'Обучение',

    // Playground
    playground_title: 'Math HWR — Площадка',
    playground_subtitle: 'Нарисуйте математическое выражение или загрузите изображение',
    brush: 'Кисть',
    eraser: 'Ластик',
    size: 'Размер',
    zoom: 'Масштаб',
    reset_view: 'Сброс',
    open_file: 'Открыть',
    clear: 'Очистить',
    recognize: 'Распознать',
    recognizing: 'Распознаём...',

    // Results
    recognized_text: 'Распознанный текст',
    copy: 'Копировать',
    copied: 'Скопировано!',
    tokens: 'Токены',
    preprocessed_image: 'Предобработанное изображение',
    inference_time: 'Время распознавания',

    // Training
    experiments: 'Эксперименты',
    new_training: 'Новое обучение',
    monitor: 'Мониторинг',
    models: 'Модели',

    // New Training
    new_training_title: 'Новый запуск обучения',
    basic_settings: 'Базовые настройки',
    advanced_settings: 'Расширенные настройки',
    model_type: 'Тип модели',
    run_name: 'Название запуска',
    batch_size: 'Размер батча',
    epochs: 'Эпохи',
    learning_rate: 'Скорость обучения',
    seed: 'Seed',
    invert_colors: 'Инверсия цветов',
    cancel: 'Отмена',
    start_training: 'Начать обучение',
    creating: 'Создание...',

    // Monitor
    status: 'Статус',
    epoch: 'Эпоха',
    loss_curves: 'Кривые потерь',
    train_loss: 'Train Loss',
    val_loss: 'Val Loss',
    latest_metrics: 'Последние метрики',
    logs: 'Логи',
    sample_predictions: 'Примеры предсказаний',
    target: 'Цель',
    pred: 'Предск.',
    match: 'Совпадение',

    // Experiments
    search_by_name: 'Поиск по имени...',
    all_models: 'Все модели',
    all_statuses: 'Все статусы',
    compare: 'Сравнить',
    no_experiments: 'Нет экспериментов',
    create_first_run: 'Создайте первый запуск обучения',
    no_matches: 'Нет экспериментов по вашим фильтрам',

    // Models
    models_checkpoints: 'Модели и чекпоинты',
    checkpoints: 'Чекпоинты',
    select_checkpoint: 'Выберите чекпоинт для просмотра деталей',
    info: 'Информация',
    quick_test: 'Быстрый тест',
    checkpoint_details: 'Детали чекпоинта',
    type: 'Тип',
    path: 'Путь',
    metrics_at_save: 'Метрики при сохранении',
    promote_to_production: 'Продвинуть в продакшн',
    download: 'Скачать',
    delete: 'Удалить',

    // Quick Test
    canvas_mode: 'Канвас',
    upload_mode: 'Загрузка',
    select_image: 'Выбрать изображение',

    // Comparison
    compare_experiments: 'Сравнение экспериментов',
    loss_curves_comparison: 'Сравнение кривых потерь',

    // Status values
    QUEUED: 'В очереди',
    RUNNING: 'Выполняется',
    FINISHED: 'Завершено',
    FAILED: 'Ошибка',
    STOPPED: 'Остановлено',

    // Messages
    loading: 'Загрузка...',
    no_logs: 'Нет логов',
    no_metrics: 'Нет метрик',
    no_checkpoints: 'Нет чекпоинтов',

    // Actions
    view: 'Просмотр',
    close: 'Закрыть',
    save: 'Сохранить',
    duplicate: 'Дублировать',
  },
};

export type TranslationKey = keyof typeof translations.en;
