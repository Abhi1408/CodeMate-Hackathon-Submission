# src/utils.py
def safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        pass
