"""Web-based desktop window — nhúng giao diện web vào desktop bằng QtWebEngine.

Desktop window này tự động khởi động web server local trên port ngẫu nhiên
và hiển thị giao diện web (HTML/CSS/JS) trong QtWebEngineView.
Giao diện desktop sẽ giống hệt giao diện web.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def find_free_port() -> int:
    """Tìm một port tự do trên localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_web_server(port: int) -> None:
    """Khởi động web server local trên port chỉ định."""
    # Import uvicorn và app từ entrypoints
    import uvicorn

    # Tắt Swagger UI trong desktop mode
    os.environ.setdefault("ONCOVISION_ENV", "desktop")

    # Import app sau khi set environment
    from entrypoints.web_app import app

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        loop="asyncio",
    )


def is_port_open(port: int) -> bool:
    """Kiểm tra web server đã mở cổng chưa."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def launch_web_desktop(
    *,
    window_title: str = "OncoVision Chat AI",
    width: int = 1480,
    height: int = 920,
) -> int:
    """Khởi động desktop window với giao diện web.

    Desktop window tự động:
    1. Tìm port tự do
    2. Khởi động web server local
    3. Mở QtWebEngineView và load giao diện web
    4. Khi đóng window, dừng web server
    """
    from PySide6.QtCore import QTimer, QUrl, Qt
    from PySide6.QtWebEngineCore import QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QProgressBar

    from app.chat_ui.paths import PROJECT_ROOT

    # Shortcut/service có thể khởi chạy exe với CWD bất kỳ; các đường dẫn
    # tương đối (config/, models/, output/) trong pipeline cần CWD = project root.
    os.chdir(PROJECT_ROOT)

    # Tìm port tự do
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    # Tạo QApplication
    app_instance = QApplication(sys.argv)
    app_instance.setApplicationName(window_title)
    app_instance.setOrganizationName("OncoVision")

    # Tạo main window
    main_window = QMainWindow()
    main_window.setWindowTitle(window_title)
    main_window.resize(width, height)

    # Tạo loading screen
    loading_widget = QWidget()
    loading_layout = QVBoxLayout(loading_widget)
    loading_layout.setAlignment(Qt.AlignCenter)

    loading_label = QLabel("Đang khởi động OncoVision...")
    loading_label.setStyleSheet("font-size: 18px; color: #666;")
    loading_layout.addWidget(loading_label)

    progress_bar = QProgressBar()
    progress_bar.setRange(0, 0)  # Indeterminate
    progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            text-align: center;
            height: 24px;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
    """)
    loading_layout.addWidget(progress_bar)

    main_window.setCentralWidget(loading_widget)
    main_window.show()

    # Tạo WebEngineView
    web_view = QWebEngineView()
    profile = QWebEngineProfile.defaultProfile()
    profile.setHttpCacheType(profile.HttpCacheType.MemoryHttpCache)

    # Khi web page load xong, chuyển từ loading sang web view
    def on_load_finished(ok: bool) -> None:
        if ok:
            # Thay thế loading widget bằng web view
            central = QWidget()
            central_layout = QVBoxLayout(central)
            central_layout.setContentsMargins(0, 0, 0, 0)
            central_layout.setSpacing(0)
            central_layout.addWidget(web_view)
            main_window.setCentralWidget(central)

    def on_title_changed(title: str) -> None:
        if title and "OncoVision" in title:
            main_window.setWindowTitle(title)

    web_view.loadFinished.connect(on_load_finished)
    web_view.titleChanged.connect(on_title_changed)

    # Khởi động web server (nạp model y khoa có thể mất ~1 phút) rồi load web
    # URL ngay khi cổng mở — polling bằng QTimer để loading screen vẫn hiển thị.
    server_thread = threading.Thread(target=start_web_server, args=(port,), daemon=True)
    server_thread.start()

    deadline = time.time() + 60.0

    def try_load() -> None:
        if is_port_open(port):
            timer.stop()
            web_view.load(QUrl(url))
        elif time.time() > deadline:
            timer.stop()
            loading_label.setText("Web server chưa sẵn sàng sau 60s. Hãy chạy lại ứng dụng.")

    timer = QTimer()
    timer.timeout.connect(try_load)
    timer.start(200)

    # Khi đóng window, thông báo (web server sẽ dừng khi thread daemon kết thúc)
    def on_close() -> None:
        # Dừng web view
        web_view.deleteLater()
        app_instance.quit()

    main_window.closeEvent = lambda e: (on_close(), e.accept())

    # Chạy application
    return app_instance.exec()


if __name__ == "__main__":
    sys.exit(launch_web_desktop(window_title="OncoVision Chat AI"))
