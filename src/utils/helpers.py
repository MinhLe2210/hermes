import io
import sys

def extract_code_block(response: str) -> str:
    """Tách phần code từ ```python ... ``` hoặc ``` ... ```"""
    import re
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1)
    return response

def run_generated_code(code: str) -> str:
    """Chạy code và trả về stdout"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code, {})  
        output = sys.stdout.getvalue()
    except Exception as e:
        output = f"⚠️ Lỗi khi chạy code: {e}"
    finally:
        sys.stdout = old_stdout
    return output