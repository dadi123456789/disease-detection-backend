# تكوين Gunicorn لتجنب timeout أثناء تحميل النموذج
timeout = 120  # 120 ثانية بدلاً من 30
workers = 1
threads = 2
worker_class = 'sync'
